#!/usr/bin/env bash
# Wipe SQLite DB and re-run migrations. Destructive: drops all jobs and artifacts metadata.
set -euo pipefail

cd "$(dirname "$0")/.."

DB_PATH="storage/pdf_translation.db"
if [[ -f "$DB_PATH" ]]; then
  echo "Removing $DB_PATH"
  rm -f "$DB_PATH"
fi

alembic upgrade head
