"""Pure, side-effect-free helpers factored out of dataset_uploader/app.py so
they're unit-testable without a real Postgres/MLflow connection -- app.py
itself does real I/O at import time (create_engine, a CREATE TABLE, reading
required env vars), which is exactly what makes it unsuitable for a fast
unit test. Same reasoning as why processing/evaluation.py is split out from
training_pipeline.py elsewhere in this codebase.

Tested in tests/test_dataset_uploader_logic.py, run in CI's fast unit_tests
job alongside test_evaluation.py/test_preprocessing.py.
"""
import re
from datetime import datetime, timedelta, timezone

from prediction_model.config import config

GMT_PLUS_1 = timezone(timedelta(hours=1))
SOURCE_LABELS = {"training": "CI/CD pipeline", "model_upload": "Direct upload"}
REQUIRED_DATASET_COLUMNS = {"timestamp", "label", "is_synthetic_anomaly", *config.METRIC_COLUMNS}


def now_gmt1():
    return datetime.now(GMT_PLUS_1)


def format_timestamp(value):
    """Renders a stored ISO timestamp (already in GMT+1 -- see now_gmt1) as a
    short, human display string. Passes through unrecognized values (e.g.
    the "unknown" default from load_dataset_metadata) unchanged.
    """
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M GMT+1")
    except (TypeError, ValueError):
        return value


def format_epoch_millis(ms):
    """Renders an MLflow ModelVersion timestamp (creation_timestamp /
    last_updated_timestamp -- epoch milliseconds, UTC) in the same short
    GMT+1 display format format_timestamp uses, so "Registered at" reads
    consistently with "Dataset at" elsewhere in the console instead of
    mixing a raw epoch int in with ISO-string fields.
    """
    if ms is None:
        return "unknown"
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(GMT_PLUS_1)
    return dt.strftime("%Y-%m-%d %H:%M GMT+1")


def format_hyperparams(params):
    """params: an MLflow run's .data.params dict. Renders every param except
    'model_type' (shown separately) as "key=value" pairs -- used to show a
    model's actual hyperparameters (e.g. LSTMAutoencoder's hidden_size/epochs)
    on the console. Uploaded models often log nothing beyond model_type (see
    training_pipeline.infer_model_type), so this returns a clear placeholder
    rather than an empty string in that case.
    """
    pairs = [f"{key}={value}" for key, value in sorted(params.items()) if key != "model_type"]
    return ", ".join(pairs) if pairs else "no hyperparameters logged"


def source_label(value):
    return SOURCE_LABELS.get(value, value)


def stage_css_class(stage):
    return {"Production": "pill-good", "Archived": "pill-neutral"}.get(stage, "pill-neutral")


def missing_dataset_columns(columns):
    """columns: an iterable of column names (e.g. a DataFrame's .columns).
    Returns the set of required columns not present -- empty set means the
    schema is valid.
    """
    return REQUIRED_DATASET_COLUMNS - set(columns)


# --------------------------------------------------------------------------
# account management (see dataset_uploader/users.py for the storage side)
# --------------------------------------------------------------------------

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
MIN_PASSWORD_LENGTH = 8


def role_label(is_admin):
    return "Administrator" if is_admin else "Member"


def validate_username(username):
    """Returns an error string, or None when the username is acceptable.
    Restricted to a conservative charset because the username is rendered in
    the console and used as the git commit author on the dataset-upload path
    (see app.py's _push_to_github).
    """
    if not username or not USERNAME_PATTERN.match(username):
        return (
            "Username must be 3-32 characters, letters/digits/dot/underscore/hyphen only."
        )
    return None


def validate_password(password):
    """Returns an error string, or None when the password is acceptable."""
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    return None


def deletion_error(actor_id, target, admin_total):
    """Whether `actor_id` may delete the `target` account.

    target: a users.list_all()-shaped dict (id, username, is_admin).
    admin_total: how many admin accounts currently exist.

    Two guards, both about not painting the console into a corner: you can't
    delete yourself (an admin mid-session removing their own account leaves a
    live session with no account behind it), and you can't remove the last
    administrator (nobody could manage accounts afterwards -- the same
    lockout users.init_schema's promotion exists to prevent).
    """
    if target["id"] == actor_id:
        return "You cannot delete your own account."
    if target["is_admin"] and admin_total <= 1:
        return "This is the last administrator -- promote another account first."
    return None


def demotion_error(actor_id, target, admin_total):
    """Whether `actor_id` may remove admin rights from `target`. Same
    last-administrator guard as deletion_error; self-demotion is allowed only
    while another admin remains.
    """
    if not target["is_admin"]:
        return None
    if admin_total <= 1:
        return "This is the last administrator -- promote another account first."
    return None


# --------------------------------------------------------------------------
# live monitoring feed (see live_feed/ for the producer side)
# --------------------------------------------------------------------------

def summarize_incidents(readings):
    """Groups consecutive readings sharing the same incident type into
    discrete events, and marks each as detected if the model flagged any
    reading inside it.

    Incident windows are recorded independently of the model's verdicts, so
    this can report detection coverage and how long detection took rather
    than only echoing back what the model claims about itself.

    readings: dicts as returned by live_feed.store.recent_readings -- 't' a
    datetime, 'incident' the incident type or None, 'is_anomaly' the model's
    verdict. Returns events oldest first.
    """
    events = []
    current = None
    for reading in readings:
        name = reading.get("incident")
        if name is None:
            current = None
            continue
        if current is None or current["name"] != name:
            current = {
                "name": name,
                "started": reading["t"],
                "ended": reading["t"],
                "readings": 0,
                "detected_at": None,
            }
            events.append(current)
        current["ended"] = reading["t"]
        current["readings"] += 1
        if reading.get("is_anomaly") and current["detected_at"] is None:
            current["detected_at"] = reading["t"]

    for event in events:
        event["detected"] = event["detected_at"] is not None
        event["detection_latency_seconds"] = (
            (event["detected_at"] - event["started"]).total_seconds()
            if event["detected_at"] is not None
            else None
        )
        event["duration_seconds"] = (event["ended"] - event["started"]).total_seconds()
    return events


# --------------------------------------------------------------------------
# overview presentation
# --------------------------------------------------------------------------

def format_age(seconds):
    """How long ago something happened, at the resolution an operator reads
    it: "4s ago", "3m ago", "2h ago", "5d ago". Negative ages (a clock a
    little ahead of the database) read as "just now" rather than "-1s ago".
    """
    if seconds is None:
        return "--"
    seconds = int(seconds)
    if seconds < 1:
        return "just now"
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit} ago"
    return f"{seconds}s ago"


def gate_position(value, threshold):
    """Geometry for the F1-against-gate meter, as percentages of its width:
    (fill, threshold marker, clears gate). Values are clamped to the 0-1 range
    F1 lives in, so a missing or malformed metric renders an empty bar rather
    than breaking the layout.
    """
    def pct(v):
        try:
            return max(0.0, min(1.0, float(v))) * 100.0
        except (TypeError, ValueError):
            return 0.0

    clears = value is not None and pct(value) >= pct(threshold)
    return pct(value) if value is not None else 0.0, pct(threshold), clears
