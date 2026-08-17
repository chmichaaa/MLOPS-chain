"""Ingestion + preparation pipeline: (re)builds dataset.csv from the NAB
source series plus the synthetic anomaly/background injection already
implemented in prediction_model/processing/build_dataset.py. Runs daily and
is manually triggerable; triggers training_pipeline right after so a fresh
dataset always gets a fresh model.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


def _run_ingestion():
    from prediction_model.processing.build_dataset import run
    run()


with DAG(
    dag_id="ingestion_pipeline",
    description="Rebuild dataset.csv from NAB sources + synthetic injection",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["infra-anomaly-detection", "ingestion"],
) as dag:
    build_dataset = PythonOperator(
        task_id="build_dataset",
        python_callable=_run_ingestion,
    )

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training_pipeline",
        wait_for_completion=False,
    )

    build_dataset >> trigger_training
