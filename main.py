from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from prediction_model.predict import generate_predictions, generate_predictions_batch
from prediction_model.config import config
import mlflow
import io
import boto3
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator


def upload_to_s3(file_content, filename):
    # endpoint_url points this at the local MinIO by default (see
    # config.MINIO_ENDPOINT_URL) -- boto3's S3 API is MinIO-compatible as-is.
    # Unset MINIO_ENDPOINT_URL to talk to real AWS S3 instead.
    s3 = boto3.client('s3', endpoint_url=config.MINIO_ENDPOINT_URL)

    current_date = datetime.now().strftime("%Y-%m-%d")
    if filename.endswith('.csv'):
        filename = filename[:-4]

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    folder_path = f"{config.FOLDER}/{current_date}"

    filename_with_datetime = f"{filename}_{current_datetime}.csv"

    s3_key = f"{folder_path}/{filename_with_datetime}"

    response = s3.put_object(Bucket=config.S3_BUCKET, Key=s3_key, Body=file_content)

    return s3_key

mlflow.set_tracking_uri(config.TRACKING_URI)

app = FastAPI(
    title="Infrastructure Anomaly Detection App using FastAPI - MLOps",
    description="MLOps Demo",
    version='1.0'
)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

Instrumentator().instrument(app).expose(app)


class MetricReading(BaseModel):
    cpu_usage_pct: float
    network_in_bytes: float
    elb_request_count: float
    rds_cpu_usage_pct: float


class MetricsWindow(BaseModel):
    readings: List[MetricReading]  # chronological order, oldest first, most recent last


@app.get("/")
def index():
    return {"message": "Welcome to the MLOps Infrastructure Anomaly Detection app"}


@app.post("/prediction_api")
def predict(window: MetricsWindow):
    readings = [r.model_dump() for r in window.readings]
    result = generate_predictions(readings)
    return result


@app.post("/prediction_ui")
def predict_gui(raw_window: str = Form(...)):
    """Manual testing helper: paste one reading per line, comma-separated, in the
    order cpu_usage_pct,network_in_bytes,elb_request_count,rds_cpu_usage_pct,
    oldest reading first.
    """
    lines = [line.strip() for line in raw_window.strip().splitlines() if line.strip()]
    if not lines:
        return {"error": "no readings provided"}

    readings = []
    for line in lines:
        values = [float(v) for v in line.split(',')]
        if len(values) != len(config.METRIC_COLUMNS):
            return {
                "error": f"each line must have {len(config.METRIC_COLUMNS)} values: "
                         f"{', '.join(config.METRIC_COLUMNS)}"
            }
        readings.append(dict(zip(config.METRIC_COLUMNS, values)))

    return generate_predictions(readings)


@app.post("/batch_prediction")
async def batch_predict(file: UploadFile = File(...)):

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content), index_col=False)

    required_columns = config.METRIC_COLUMNS
    if not all(column in df.columns for column in required_columns):
        return {"error": "CSV file does not contain the required columns."}

    predictions = generate_predictions_batch(df)

    df['anomaly_score'] = predictions["anomaly_score"]
    df['is_anomaly'] = predictions["is_anomaly"]
    result = df.to_csv(index=False)

    s3_key = upload_to_s3(result.encode('utf-8'), file.filename)

    return StreamingResponse(io.BytesIO(result.encode('utf-8')), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
