"""Trains against the synthetic "easy" dataset (see
processing/build_synthetic_dataset.py) to validate the promotion gate itself
-- NOT to measure real-world model quality; that stays the NAB pipeline's job
(training_pipeline.py, unaffected by this script).

Runs the exact same Hyperopt search + gate logic as the NAB pipeline
(train_and_select, shared from training_pipeline.py), logged to a SEPARATE
MLflow experiment (config.SYNTHETIC_EASY_EXPERIMENT_NAME) against the
ORIGINAL spec threshold (config.SYNTHETIC_EASY_F1_THRESHOLD = 0.85), which
real NAB data cannot honestly reach with this model class. A pass here
demonstrates the gate correctly promotes a good model when the data actually
supports it.

    python -m prediction_model.processing.build_synthetic_dataset  # once, or after config changes
    python -m prediction_model.train_synthetic_easy
"""
from hyperopt import hp

from prediction_model.config import config
from prediction_model.processing.data_handling import load_synthetic_easy_dataset
from prediction_model.training_pipeline import train_and_select

# Narrower than the NAB search space's contamination/nu floor (0.05): this
# dataset's true anomaly rate is 60/2000 = 3%, so a 5% floor would force every
# trial to flag more points than actually exist as anomalous, capping
# precision regardless of how separable the anomalies are. Passed explicitly
# (see train_and_select's docstring) rather than edited in training_pipeline.py
# so NAB's own search space -- and its already-verified result -- is untouched.
isolation_forest_space = {
    'n_estimators': hp.choice('if_n_estimators', [50, 100, 150, 200, 300]),
    'max_features': hp.uniform('if_max_features', 0.5, 1.0),
    'contamination': hp.uniform('if_contamination', 0.01, 0.1),
    'bootstrap': hp.choice('if_bootstrap', [True, False]),
}

one_class_svm_space = {
    'nu': hp.uniform('ocsvm_nu', 0.01, 0.1),
    'gamma': hp.choice('ocsvm_gamma', ['scale', 'auto']),
}

# NAB's WINDOW_SIZE=12 rolling-mean feature is tuned for multi-hour sustained
# incidents. Applied to isolated single-sample spikes, the rolling mean smears
# each spike across its trailing 11 samples, contaminating nearby normal
# points' features and capping precision regardless of separability (verified:
# F1 rose from ~0.25 at window=12 to ~0.9 at window=1 on identical data). A
# single-point anomaly has no rolling context to exploit, so window=1 -- not
# NAB's window -- is the methodologically correct choice here, not a tuning hack.
SYNTHETIC_EASY_WINDOW_SIZE = 1

if __name__ == "__main__":
    train_df, eval_df = load_synthetic_easy_dataset()
    X_train = train_df[config.METRIC_COLUMNS]

    train_and_select(
        X_train=X_train,
        eval_df=eval_df,
        experiment_name=config.SYNTHETIC_EASY_EXPERIMENT_NAME,
        dataset_type='synthetic_easy',
        f1_threshold=config.SYNTHETIC_EASY_F1_THRESHOLD,
        isolation_forest_space=isolation_forest_space,
        one_class_svm_space=one_class_svm_space,
        window_size=SYNTHETIC_EASY_WINDOW_SIZE,
    )
