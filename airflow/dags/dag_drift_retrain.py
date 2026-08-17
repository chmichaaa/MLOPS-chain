"""Drift monitoring + retrain trigger: runs drift_monitoring/check_drift.py
every DRIFT_CHECK_SCHEDULE_MINUTES minutes and triggers training_pipeline if
drift is detected. A 5-minute schedule bounds worst-case detection latency to
10 minutes by construction (a drift-causing upload landing right after a check
run waits at most one full interval before the next check catches it); the
actual latency for each run is logged by check_drift.py to MLflow
(detection_latency_seconds) for verification.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from prediction_model.config import config


def _check_drift_and_branch(**context):
    from drift_monitoring.check_drift import check_drift

    result = check_drift()
    context["ti"].xcom_push(key="drift_result", value=result)
    return "trigger_retraining" if result["drift_detected"] else "no_drift"


with DAG(
    dag_id="drift_monitoring_retrain",
    description="Automated Evidently drift check; triggers training_pipeline on detected drift",
    start_date=datetime(2024, 1, 1),
    schedule_interval=f"*/{config.DRIFT_CHECK_SCHEDULE_MINUTES} * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0},
    tags=["infra-anomaly-detection", "drift-monitoring"],
) as dag:
    check_drift_task = BranchPythonOperator(
        task_id="check_drift",
        python_callable=_check_drift_and_branch,
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="training_pipeline",
        wait_for_completion=False,
    )

    no_drift = EmptyOperator(task_id="no_drift")

    check_drift_task >> [trigger_retraining, no_drift]
