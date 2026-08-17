"""CI helper: reports whether the best run in an MLflow experiment meets a
given F1 threshold -- WITHOUT itself failing. Writes a human-readable summary
to $GITHUB_STEP_SUMMARY (shown directly in the GitHub Actions UI) and sets a
`meets_threshold` step output.

Deliberately never fails: this is what lets validate_nab_real (whose own real
gate is config.F1_THRESHOLD = 0.58, unchanged) also report -- visibly, without
turning the job red -- whether it clears the ORIGINAL spec's 0.85 bar, which
it doesn't and isn't expected to (see config.py's F1_THRESHOLD comment). The
job's pass/fail should reflect its own real gate; whether it separately also
clears 0.85 is informational. Conflating "the job ran correctly" with "the
model hit an unrelated number" is what this script exists to avoid.

    python prediction_model/report_gate_status.py \\
        --experiment infra_anomaly_detection --threshold 0.85 \\
        --label "NAB real (original spec bar)"
"""
import argparse
import os

import mlflow

from prediction_model.config import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        print(f"No MLflow experiment '{args.experiment}' found -- nothing to report.")
        return

    runs_df = mlflow.search_runs(experiment_ids=experiment.experiment_id, order_by=['metrics.f1_score DESC'])
    if runs_df.empty:
        print(f"No runs logged yet in '{args.experiment}' -- nothing to report.")
        return

    best_f1 = runs_df.iloc[0]['metrics.f1_score']
    meets = bool(best_f1 >= args.threshold)

    summary = (
        f"### {args.label}\n"
        f"- Experiment: `{args.experiment}`\n"
        f"- Best F1 (pointwise): **{best_f1:.4f}**\n"
        f"- Threshold checked: {args.threshold}\n"
        f"- Meets threshold: **{'YES' if meets else 'NO'}**\n"
    )
    print(summary)

    step_summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if step_summary_path:
        with open(step_summary_path, 'a') as f:
            f.write(summary + "\n")

    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"meets_threshold={'true' if meets else 'false'}\n")


if __name__ == "__main__":
    main()
