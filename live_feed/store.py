"""Shared schema and access helpers for the live monitoring feed.

live_feed/generator.py writes here; dataset_uploader's /live page reads from
here. Kept in one module so there is exactly one definition of the table and
one place that knows its shape, rather than a CREATE TABLE in the writer and
hand-rolled SELECTs in the reader that drift apart over time.

The table lives in the existing dataset_uploader Postgres database (created by
docker/postgres-init.sh) rather than a new one: the console is its only
consumer, and these rows are demo/monitoring state, not model artifacts --
losing them costs nothing, which is also why prune_older_than exists rather
than any archival story.
"""
import os

from sqlalchemy import create_engine, text

from prediction_model.config import config

_engine = None


def engine():
    """One lazily-created engine per process. pool_pre_ping because the
    generator is a long-running loop: without it, every Postgres restart
    leaves it holding dead connections and raising on the next insert.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    return _engine


def init_schema():
    with engine().begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS live_readings (
                    id BIGSERIAL PRIMARY KEY,
                    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    cpu_usage_pct DOUBLE PRECISION NOT NULL,
                    network_in_bytes DOUBLE PRECISION NOT NULL,
                    elb_request_count DOUBLE PRECISION NOT NULL,
                    rds_cpu_usage_pct DOUBLE PRECISION NOT NULL,
                    anomaly_score DOUBLE PRECISION,
                    is_anomaly BOOLEAN,
                    model_version TEXT,
                    injected_incident TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS live_readings_observed_at_idx "
                "ON live_readings (observed_at DESC)"
            )
        )


def insert_reading(reading, verdict, injected_incident):
    """reading: {metric: value} for config.METRIC_COLUMNS.
    verdict: the /prediction_api response, or None when scoring failed (the
    app is down, or no model is in Production yet) -- the reading is still
    stored so the chart shows a real gap instead of silently skipping time.
    injected_incident: the scenario name this reading was generated under,
    or None for normal traffic. This is the ground truth the console's /live
    page grades the model against.
    """
    row = dict(reading)
    row["anomaly_score"] = verdict.get("anomaly_score") if verdict else None
    row["is_anomaly"] = verdict.get("is_anomaly") if verdict else None
    row["model_version"] = str(verdict.get("model_version")) if verdict and verdict.get("model_version") else None
    row["injected_incident"] = injected_incident

    with engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO live_readings (
                    cpu_usage_pct, network_in_bytes, elb_request_count, rds_cpu_usage_pct,
                    anomaly_score, is_anomaly, model_version, injected_incident
                ) VALUES (
                    :cpu_usage_pct, :network_in_bytes, :elb_request_count, :rds_cpu_usage_pct,
                    :anomaly_score, :is_anomaly, :model_version, :injected_incident
                )
                """
            ),
            row,
        )


def recent_readings(limit=180):
    """Returns the newest `limit` readings, oldest first -- chart order, so
    callers never have to remember to reverse it.
    """
    with engine().begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT observed_at, cpu_usage_pct, network_in_bytes, elb_request_count,
                       rds_cpu_usage_pct, anomaly_score, is_anomaly, model_version, injected_incident
                FROM live_readings
                ORDER BY observed_at DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        ).mappings().all()

    readings = []
    for row in reversed(rows):
        reading = {"t": row["observed_at"], "incident": row["injected_incident"]}
        for metric in config.METRIC_COLUMNS:
            reading[metric] = row[metric]
        reading["anomaly_score"] = row["anomaly_score"]
        reading["is_anomaly"] = row["is_anomaly"]
        reading["model_version"] = row["model_version"]
        readings.append(reading)
    return readings


def prune_older_than(hours):
    with engine().begin() as conn:
        conn.execute(
            text("DELETE FROM live_readings WHERE observed_at < now() - make_interval(hours => :hours)"),
            {"hours": int(hours)},
        )
