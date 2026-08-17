"""Programmatic drift check. Meant to be run frequently by
airflow/dags/dag_drift_retrain.py: compares the latest batch-prediction upload
against the training reference and, if drift exceeds config.DRIFT_THRESHOLD,
signals that a retrain is warranted.

Also logs `detection_latency_seconds` = (time of this check) - (upload time of
the newest batch file examined) to MLflow, which is the concrete number that
demonstrates the project's "<10 min detection delay" requirement: run on a
DRIFT_CHECK_SCHEDULE_MINUTES-minute schedule (see dag_drift_retrain.py), the
worst case is bounded by construction to 2x that interval, and this metric
shows what it actually was for a given run.
"""
import io
from datetime import datetime, timezone

import boto3
import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from prediction_model.config import config
from prediction_model.processing.data_handling import load_full_dataset, day_block_split


def _s3_client():
    return boto3.client("s3", endpoint_url=config.MINIO_ENDPOINT_URL)


def _latest_batch_object(s3):
    response = s3.list_objects_v2(Bucket=config.S3_BUCKET, Prefix=f"{config.FOLDER}/")
    contents = response.get("Contents", [])
    if not contents:
        return None
    return max(contents, key=lambda obj: obj["LastModified"])


def _load_reference():
    dataset = load_full_dataset()
    train_df, _ = day_block_split(dataset)
    return train_df[config.METRIC_COLUMNS]


def check_drift():
    s3 = _s3_client()
    latest = _latest_batch_object(s3)
    if latest is None:
        print("No batch-prediction uploads found yet -- nothing to compare against.")
        return {"drift_detected": False, "reason": "no_data"}

    body = s3.get_object(Bucket=config.S3_BUCKET, Key=latest["Key"])["Body"].read()
    current = pd.read_csv(io.BytesIO(body))[config.METRIC_COLUMNS]
    reference = _load_reference()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()["metrics"][0]["result"]

    share_drifted = result["share_of_drifted_columns"]
    dataset_drift = result["dataset_drift"]
    drift_detected = dataset_drift or share_drifted >= config.DRIFT_THRESHOLD

    check_time = datetime.now(timezone.utc)
    upload_time = latest["LastModified"]
    detection_latency_seconds = (check_time - upload_time).total_seconds()

    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment("drift_monitoring")
    with mlflow.start_run():
        mlflow.log_metrics(
            {
                "share_of_drifted_columns": share_drifted,
                "number_of_drifted_columns": result["number_of_drifted_columns"],
                "dataset_drift": float(dataset_drift),
                "detection_latency_seconds": detection_latency_seconds,
            }
        )
        mlflow.log_param("checked_file", latest["Key"])

    print(
        f"checked {latest['Key']}: share_of_drifted_columns={share_drifted:.2f} "
        f"dataset_drift={dataset_drift} detection_latency_seconds={detection_latency_seconds:.0f} "
        f"-> drift_detected={drift_detected}"
    )
    return {"drift_detected": drift_detected, "share_of_drifted_columns": share_drifted}


if __name__ == "__main__":
    check_drift()
