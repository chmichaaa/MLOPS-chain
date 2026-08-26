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

# boto3 pinned to the exact version requirements.txt/Dockerfile.mlflow already
# use elsewhere in this project -- left unpinned, pip's resolver backtracks
# through dozens of old aiobotocore releases (a transitive dep of dvc[s3]'s
# s3fs) hunting for one whose 'boto3' extra matches, which is extremely slow.
#
# bcrypt pinned below 4.1: passlib (last released 2020, unmaintained) does an
# internal self-test against bcrypt's 72-byte input limit at import/first-use
# time; bcrypt>=4.1 changed that limit from "silently truncate" to "raise
# ValueError", which crashes passlib's own self-test on every hash/verify
# call. 4.0.1 is the last version with the old, passlib-compatible behavior.
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] python-multipart itsdangerous "passlib[bcrypt]" "bcrypt==4.0.1" \
    sqlalchemy psycopg2-binary \
    pandas scikit-learn hyperopt mlflow==2.13.2 cloudpickle boto3==1.34.144 "dvc[s3]"

ENV PYTHONPATH=/repo
WORKDIR /repo

EXPOSE 8090

CMD ["uvicorn", "dataset_uploader.app:app", "--host", "0.0.0.0", "--port", "8090"]
