"""Training pipeline: runs the Hyperopt search over Isolation Forest / OCSVM
(prediction_model/training_pipeline.py, unchanged from Priority 1) and logs
every trial to MLflow. Uses BashOperator rather than importing the module
directly -- training_pipeline.py runs its search at import time (same pattern
the original repo used for this file) and asserts the F1 threshold, so a
subprocess with a real exit code is the natural fit: a failing assert fails
the task, which is exactly the "don't promote a bad model" behaviour wanted
here.

Triggered manually, by dag_drift_retrain.py on detected drift, or right after
dag_ingestion.py (schedule_interval=None -- not run on its own clock, since it
should always operate on a freshly built dataset.csv rather than a stale one).
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="training_pipeline",
    description="Hyperopt search (Isolation Forest / OCSVM) + MLflow logging",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # triggered only -- see module docstring
    catchup=False,
    default_args={"retries": 0},
    tags=["infra-anomaly-detection", "training"],
) as dag:
    train = BashOperator(
        task_id="train",
        bash_command="cd /opt/airflow/project && python -m prediction_model.training_pipeline",
    )
