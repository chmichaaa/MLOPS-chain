"""Daily backup: pg_dump every Postgres database + mirror the MinIO bucket to
a REAL AWS S3 bucket -- not MinIO. Protects against losing the EC2 instance
itself (disk failure, accidental termination), not just a container restart,
since everything else in this stack (MLflow history, the model registry,
dataset_uploader's user accounts, DVC data) otherwise lives only on that one
box's Docker volumes.

Needs a separate, real AWS IAM user (BACKUP_AWS_ACCESS_KEY_ID/SECRET) scoped
to just the one backup bucket -- deliberately not the AWS_ACCESS_KEY_ID/
SECRET already used everywhere else in this stack, which are MinIO's
credentials, not real AWS ones. If those two env vars aren't set, both tasks
fail fast with a clear error rather than silently doing nothing.

Full mirror of the MinIO bucket every run, not incremental -- the bucket is
small at this project's scale, so trading a little wasted transfer for much
simpler code is the right call here.
"""
import os
import subprocess
from datetime import datetime, timedelta, timezone

import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator

DATABASES = ["mlflow", "airflow", "dataset_uploader"]
GMT_PLUS_1 = timezone(timedelta(hours=1))


def _backup_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["BACKUP_AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["BACKUP_AWS_SECRET_ACCESS_KEY"],
    )


def _dump_and_upload_databases():
    s3 = _backup_client()
    bucket = os.environ["BACKUP_S3_BUCKET"]
    date_prefix = datetime.now(GMT_PLUS_1).strftime("%Y-%m-%d")

    for db in DATABASES:
        dump_path = f"/tmp/{db}.sql"
        subprocess.run(
            ["pg_dump", "-h", "postgres", "-U", os.environ["POSTGRES_USER"], "-d", db, "-f", dump_path],
            env={**os.environ, "PGPASSWORD": os.environ["POSTGRES_PASSWORD"]},
            check=True,
        )
        s3.upload_file(dump_path, bucket, f"backups/{date_prefix}/postgres/{db}.sql")
        os.remove(dump_path)


def _mirror_minio_bucket():
    minio = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3 = _backup_client()
    bucket = os.environ["BACKUP_S3_BUCKET"]
    minio_bucket = os.environ.get("S3_BUCKET", "infra-monitoring")
    date_prefix = datetime.now(GMT_PLUS_1).strftime("%Y-%m-%d")

    paginator = minio.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=minio_bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            body = minio.get_object(Bucket=minio_bucket, Key=key)["Body"].read()
            s3.put_object(Bucket=bucket, Key=f"backups/{date_prefix}/minio/{key}", Body=body)


with DAG(
    dag_id="backup",
    description="Daily: pg_dump every Postgres DB + mirror the MinIO bucket to real AWS S3",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=15)},
    tags=["infra-anomaly-detection", "backup"],
) as dag:
    backup_databases = PythonOperator(task_id="backup_databases", python_callable=_dump_and_upload_databases)
    backup_minio = PythonOperator(task_id="backup_minio", python_callable=_mirror_minio_bucket)
