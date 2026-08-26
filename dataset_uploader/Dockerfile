# A separate admin console, deliberately not sharing the main app's image:
# this one needs git + a GitHub push token + write access to the real repo
# checkout, none of which belong in the slim, CI-rebuilt `app` image. Project
# code (including this app itself) is NOT copied in -- docker-compose.yml
# bind-mounts the real repo checkout at /repo, same pattern as the airflow
# service, so it always runs against the actual working tree.
# Not -buster like the other Dockerfiles in this repo: Debian buster is EOL and
# its apt mirrors (deb.debian.org) 404 on fresh builds now. This image is new,
# so it just targets a still-supported Debian release instead of needing an
# archive.debian.org mirror workaround.
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] python-multipart itsdangerous "passlib[bcrypt]" \
    sqlalchemy psycopg2-binary \
    pandas scikit-learn hyperopt mlflow==2.13.2 cloudpickle boto3 "dvc[s3]"

ENV PYTHONPATH=/repo
WORKDIR /repo

EXPOSE 8090

CMD ["uvicorn", "dataset_uploader.app:app", "--host", "0.0.0.0", "--port", "8090"]
