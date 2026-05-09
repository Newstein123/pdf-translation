#!/usr/bin/env bash
# Start API + Celery worker together for local dev. Assumes Redis is running via docker-compose.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo ".env not found. Copying from .env.example. Set GEMINI_API_KEY before using translation."
  cp .env.example .env
fi

alembic upgrade head

uvicorn app.main:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}" --reload &
API_PID=$!

celery -A app.workers.celery_app.celery_app worker --loglevel=info --concurrency="${TRANSLATION_CONCURRENCY:-5}" &
WORKER_PID=$!

trap 'kill $API_PID $WORKER_PID 2>/dev/null || true' EXIT
wait
