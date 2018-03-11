#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE USER pytech WITH PASSWORD 'pytech';
    CREATE USER test WITH PASSWORD 'test';
    ALTER USER pytech WITH SUPERUSER;
    ALTER USER test WITH SUPERUSER;
    CREATE DATABASE pytech;
    CREATE DATABASE test;
    GRANT ALL PRIVILEGES ON DATABASE pytech TO pytech;
    GRANT ALL PRIVILEGES ON DATABASE test TO pytech;
    SET timescaledb.allow_install_without_preload = 'on';
EOSQL
