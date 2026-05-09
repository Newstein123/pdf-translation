# PDF → Burmese Translation

Local single-user tool that translates foreign-language PDFs (up to 500 pages) into Burmese using Gemini AI, preserving layout and producing both PDF and EPUB outputs.

**Status:** M11 — Production foundation in flight. Pipeline is end-to-end (upload → extract → classify → multi-pass translate → rebuild PDF + EPUB). Quality and cost work tracked across three parallel tracks.

---

## Prerequisites

- Python 3.11+
- Redis (via Docker: `docker compose up -d redis`)
- Tesseract for scanned-PDF OCR: `brew install tesseract tesseract-lang` on macOS
- A Gemini API key (https://aistudio.google.com/apikey)

## Daily run (returning to a working setup)

```sh
source .venv/bin/activate
make up        # starts Redis + migrations + API + Celery worker
```

Ctrl+C stops the API and Celery worker. To also stop the Redis container:

```sh
make stop      # kill API + worker + Redis
make restart   # stop + start in one step
make test      # run full pytest suite
```

Run `make help` to see all targets.

Then open **http://127.0.0.1:8000** in a browser, paste your `API_BEARER_TOKEN` (from `.env`), upload a PDF.

## URLs once running

| What | URL |
|---|---|
| **Web UI** (upload, run, download) | http://127.0.0.1:8000 |
| Translation history | http://127.0.0.1:8000/history |
| Source ↔ target compare view | http://127.0.0.1:8000/compare |
| Health check | http://127.0.0.1:8000/health |
| Interactive API docs (Swagger) | http://127.0.0.1:8000/docs |
| ReDoc API docs | http://127.0.0.1:8000/redoc |

Default port is **8000**. To change it, set `API_PORT` in `.env` (the dev script reads `${API_PORT:-8000}`).

## First-time setup

```sh
make install                    # create venv, install deps, copy .env.example, fetch fonts
source .venv/bin/activate
# edit .env — set API_BEARER_TOKEN and GEMINI_API_KEY
make up
```

`make install` is a one-shot helper; after that, `make up` is all you need to boot the stack.

Health check:

```sh
curl http://127.0.0.1:8000/health
```

For the full setup walkthrough, troubleshooting, and curl recipes for the API surface, see [GETTING_STARTED.md](GETTING_STARTED.md).

## API (M1 surface)

All endpoints require `Authorization: Bearer <API_BEARER_TOKEN>`.

**Upload a PDF**

```sh
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H "Authorization: Bearer $API_BEARER_TOKEN" \
  -F "file=@sample.pdf"
```

**Check status**

```sh
curl http://127.0.0.1:8000/api/jobs/<job_id> \
  -H "Authorization: Bearer $API_BEARER_TOKEN"
```

**Delete a job + artifacts**

```sh
curl -X DELETE http://127.0.0.1:8000/api/jobs/<job_id> \
  -H "Authorization: Bearer $API_BEARER_TOKEN"
```

## Layout

See the plan at `/Users/applegarden/.claude/plans/ok-makes-plan-replicated-abelson.md` for the full architecture.

```
app/
├── api/          # FastAPI routes + auth
├── db/           # SQLAlchemy models, session, Alembic migrations
├── schemas/      # Pydantic DTOs
├── services/     # business logic (jobs today; extraction/translation in M2+)
├── workers/      # Celery app + stage tasks
├── prompts/      # Gemini prompt templates (M3)
└── utils/        # fonts, tokens, checkpoints (M3+)
```

## Tests

```sh
pytest
```

## Reset local state

```sh
./scripts/reset_db.sh
```
