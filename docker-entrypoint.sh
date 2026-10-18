#!/bin/sh
set -e

case "$1" in
    web)
        echo "Starting Prism Review web server (uvicorn)..."
        exec uvicorn prism.api:app --host 0.0.0.0 --port 8000
        ;;
    worker)
        echo "Starting Prism Review async worker (arq)..."
        exec arq prism.queue.worker.WorkerSettings
        ;;
    migrate)
        echo "Running database migrations (alembic upgrade head)..."
        exec alembic upgrade head
        ;;
    *)
        exec "$@"
        ;;
esac
