# Extends the official Airflow image with this project's dependencies, so DAGs
# can `import prediction_model...` / `import drift_monitoring...` directly.
# Project code itself is NOT copied in -- docker-compose.yml mounts the repo
# root at /opt/airflow/project (PYTHONPATH) so code changes don't need a rebuild.
#
# Note: this installs requirements.txt's pins (numpy/pandas/scikit-learn/etc.)
# on top of whatever Airflow 2.9 already depends on. Airflow's own dependency
# set is usually permissive enough for this to resolve cleanly, but if pip
# reports a conflict, loosen the offending pin in requirements.txt rather than
# fighting Airflow's constraints file.
FROM apache/airflow:2.9.3-python3.10

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
