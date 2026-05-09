# PDF translation — local dev workflow.
#
# Most-used commands:
#   make up        Start Redis, apply migrations, run API + Celery worker (Ctrl+C to stop)
#   make stop      Stop Redis container and any leftover API / worker processes
#   make restart   stop + up
#   make test      Run the full pytest suite
#
# All commands assume you've activated the venv:
#   source .venv/bin/activate
# (run ``make install`` once after cloning to set it up).

PYTHON ?= python3
VENV   ?= .venv

.DEFAULT_GOAL := up
.PHONY: help up start stop restart logs test migrate redis-up redis-down install clean

help:  ## show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up: redis-up migrate  ## start everything (redis, migrations, API + Celery worker; foreground)
	@echo ""
	@echo "→ http://127.0.0.1:8000 — paste your API_BEARER_TOKEN to begin"
	@echo "→ Ctrl+C to stop API + worker (Redis stays up; run 'make stop' to kill that too)"
	@echo ""
	@./scripts/dev.sh

start: up  ## alias for ``make up``

stop:  ## stop the API, the Celery worker, and the Redis container
	-@pkill -f "uvicorn app.main:app" 2>/dev/null && echo "stopped uvicorn" || true
	-@pkill -f "celery -A app.workers" 2>/dev/null && echo "stopped celery" || true
	-@docker compose down 2>/dev/null && echo "stopped redis" || true
	@echo "all stopped."

restart: stop up  ## stop + up

redis-up:  ## start the Redis container if not already running
	@docker compose up -d redis 2>/dev/null

redis-down:  ## stop the Redis container only
	@docker compose down

migrate:  ## apply pending Alembic migrations
	@alembic upgrade head

test:  ## run the full pytest suite
	@pytest -q

logs:  ## follow Redis logs (uvicorn + celery logs are in your foreground terminal)
	@docker compose logs -f redis

install:  ## first-time setup (venv, deps, .env, fonts)
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -e '.[dev]'
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env — set API_BEARER_TOKEN and GEMINI_API_KEY before ``make up``"; fi
	@bash scripts/download_fonts.sh
	@echo ""
	@echo "Setup complete. Activate the venv then ``make up``:"
	@echo "  source $(VENV)/bin/activate"
	@echo "  make up"

clean:  ## stop everything and remove build artifacts (caches, etc.)
	@$(MAKE) stop
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned."
