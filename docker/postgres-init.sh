#!/bin/bash
# Runs once on first container start (postgres image convention: any script in
# /docker-entrypoint-initdb.d/ is executed automatically). Creates the logical
# databases this stack needs on a single Postgres instance -- one for MLflow's
# backend store, one for Airflow's metadata DB, one for dataset_uploader's
# user accounts.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    CREATE DATABASE mlflow;
    CREATE DATABASE airflow;
    CREATE DATABASE dataset_uploader;
EOSQL
