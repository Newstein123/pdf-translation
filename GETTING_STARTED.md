# Getting Started

This guide walks you from a fresh clone to a working M1 stack you can hit with `curl`. Follow top to bottom.

---

## 1. What you need to install

### System dependencies

| Tool | Why | Install (macOS) |
|---|---|---|
| Python 3.11+ | Runtime | `brew install python@3.12` (already on system) |
| Docker Desktop | Run Redis locally | https://www.docker.com/products/docker-desktop |
| Tesseract + Burmese pack | OCR fallback (used in M2) | `brew install tesseract tesseract-lang` |

> Tesseract isn't required to run M1 (just upload + status). Install it before M2.

### Optional, recommended

- **direnv** to auto-load `.env`: `brew install direnv`
- **httpie** for nicer API testing: `brew install httpie`

---

## 2. What you need to add (accounts / keys)

### Required for M3+ (translation)

- **Gemini API key** — get one at https://aistudio.google.com/app/apikey. Free tier is enough for development. Paste into `.env` as `GEMINI_API_KEY=...`.

### Required from day one

- **API_BEARER_TOKEN** — pick any random string. This is the bearer token every API call must send. Generate one with:

  ```sh
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

That's it. No other accounts, no cloud setup. Everything runs on your machine.

---

## 3. First-time setup (one-time commands)

Run these from the project root: `/Users/applegarden/Desktop/AstroDev/pdf-translation`.

```sh
# 1. Create venv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# 2. Create your .env from the template
cp .env.example .env

# 3. Edit .env and set the two values you need today:
#      API_BEARER_TOKEN=<paste your generated token>
#      GEMINI_API_KEY=<paste later, when you reach M2/M3>

# 4. Start Redis (in the background)
docker compose up -d redis

# 5. Apply DB schema (creates storage/pdf_translation.db)
alembic upgrade head
```

Verify:

```sh
docker ps              # should show pdf-translation-redis-1 running
ls storage/            # should show pdf_translation.db, uploads/, intermediate/, outputs/
```

---

## 4. Running the app

The dev script starts the API and a Celery worker together. Redis must already be running (step 4 above).

```sh
source .venv/bin/activate
./scripts/dev.sh
```

You should see Uvicorn boot on `http://127.0.0.1:8000` and Celery print a banner with the four registered tasks (`jobs.extract`, `jobs.classify`, `jobs.translate`, `jobs.rebuild`).

> If you'd rather run them separately (e.g. for cleaner logs), open two terminals:
>
> ```sh
> # terminal 1
> uvicorn app.main:app --reload
>
> # terminal 2
> celery -A app.workers.celery_app.celery_app worker --loglevel=info --concurrency=5
> ```

---

## 5. Use it

### Web UI (easiest)

Open **http://127.0.0.1:8000** in your browser.

1. Paste your `API_BEARER_TOKEN` and click **Save** (stored in localStorage only).
2. Pick a PDF and click **Upload**.
3. Wait for status `AWAITING_CONFIRMATION` — the classifier will suggest NOVEL or EDUCATIONAL.
4. Confirm the category, optionally upload a glossary, click **Start translation**.
5. Watch live progress (SSE). When status hits `COMPLETED`, download links for the PDF + EPUB appear.

### curl (alternative)

```sh
export TOKEN="<the value of API_BEARER_TOKEN from .env>"

# Health check
curl http://127.0.0.1:8000/health

# Upload a PDF (replace path)
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/some.pdf"

# Check status (use the id from the upload response)
curl http://127.0.0.1:8000/api/jobs/<job_id> \
  -H "Authorization: Bearer $TOKEN"

# Confirm category (or override)
curl -X PATCH http://127.0.0.1:8000/api/jobs/<job_id>/category \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"category": "NOVEL"}'

# (optional) Upload glossary
curl -X POST http://127.0.0.1:8000/api/jobs/<job_id>/glossary \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@glossary.json"

# Start translation
curl -X POST http://127.0.0.1:8000/api/jobs/<job_id>/start \
  -H "Authorization: Bearer $TOKEN"

# Watch live events
curl -N "http://127.0.0.1:8000/api/jobs/<job_id>/events?token=$TOKEN"

# Cancel an in-flight job
curl -X POST http://127.0.0.1:8000/api/jobs/<job_id>/cancel \
  -H "Authorization: Bearer $TOKEN"

# Resume a cancelled or crashed job
curl -X POST http://127.0.0.1:8000/api/jobs/<job_id>/resume \
  -H "Authorization: Bearer $TOKEN"

# Download outputs
curl -O -J "http://127.0.0.1:8000/api/jobs/<job_id>/outputs/pdf?token=$TOKEN"
curl -O -J "http://127.0.0.1:8000/api/jobs/<job_id>/outputs/epub?token=$TOKEN"
```

### Maintenance

```sh
# Preview what would be purged (no deletes):
curl -X POST "http://127.0.0.1:8000/api/admin/purge?dry_run=true" \
  -H "Authorization: Bearer $TOKEN"

# Actually delete artifacts older than ARTIFACT_RETENTION_HOURS:
curl -X POST "http://127.0.0.1:8000/api/admin/purge" \
  -H "Authorization: Bearer $TOKEN"
```

What you should see:
- Upload returns `201` with `{"id": "...", "status": "UPLOADED", ...}`.
- Status returns the same job, with `status` flipping to `EXTRACTING` shortly after (the worker picks up the stub task).
- The PDF lands at `storage/uploads/<job_id>.pdf`.
- Job rows + a `JobEvent` row exist in `storage/pdf_translation.db`.

Inspect the DB if you want:

```sh
sqlite3 storage/pdf_translation.db "SELECT id, status, page_count, original_filename FROM jobs;"
```

---

## 6. What works vs. what's stubbed

| Stage | Status | Where |
|---|---|---|
| Upload + validation | ✅ M1 | [app/api/routes_jobs.py](app/api/routes_jobs.py) |
| Job tracking (DB + events) | ✅ M1 | [app/services/jobs.py](app/services/jobs.py) |
| Bearer auth (header + query) | ✅ M1/M5 | [app/api/auth.py](app/api/auth.py) |
| Celery wiring | ✅ M1 | [app/workers/celery_app.py](app/workers/celery_app.py) |
| Extraction (PyMuPDF + OCR) | ✅ M2 | [app/services/extraction.py](app/services/extraction.py) |
| Classification (Gemini) | ✅ M2 | [app/services/classification.py](app/services/classification.py) |
| Chunking + translation | ✅ M3 | [app/services/chunking.py](app/services/chunking.py) [translation.py](app/services/translation.py) |
| Glossary upload + enforcement | ✅ M3 | [app/services/glossary.py](app/services/glossary.py) |
| Name registry | ✅ M3 | [app/services/jobs.py](app/services/jobs.py) |
| PDF rebuild | ✅ M4 | [app/services/rebuild_pdf.py](app/services/rebuild_pdf.py) |
| EPUB rebuild | ✅ M4 | [app/services/rebuild_epub.py](app/services/rebuild_epub.py) |
| SSE progress stream | ✅ M5 | [app/api/routes_events.py](app/api/routes_events.py) |
| Web UI | ✅ M5 | [web/](web/) |
| Cancel + resume | ✅ M6 | [app/api/routes_jobs.py](app/api/routes_jobs.py) |
| Cost reporting (tokens/job) | ✅ M6 | [app/services/gemini.py](app/services/gemini.py) → `total_tokens` in status |
| Retention / purge admin | ✅ M6 | [app/api/routes_admin.py](app/api/routes_admin.py) |
| Gemini retry + backoff | ✅ M6 | [app/services/gemini.py](app/services/gemini.py) |
| 500-page benchmark | ⏳ optional | run a real PDF; metrics in `job_events` |

---

## 7. Running tests

```sh
source .venv/bin/activate
pytest
```

Five integration tests should pass.

---

## 8. Resetting state

Wipe the SQLite DB + re-run migrations (does **not** delete uploaded PDFs):

```sh
./scripts/reset_db.sh
```

To also clear PDFs and intermediate files:

```sh
rm -rf storage/uploads/* storage/intermediate/* storage/outputs/*
```

To stop Redis:

```sh
docker compose down
```

---

## 9. Common gotchas

- **`401 invalid bearer token`** — your `Authorization` header doesn't match `API_BEARER_TOKEN` in `.env`. Either you forgot to source the env, or the worker process and the API process were started with different `.env` values.
- **`Connection refused` to Redis** — Docker isn't running, or the container died. Run `docker compose up -d redis` again.
- **Worker doesn't seem to pick up tasks** — make sure Redis is on `localhost:6379`. Check `CELERY_BROKER_URL` in `.env`.
- **`alembic` can't find the DB** — confirm you're in the project root; `DATABASE_URL` defaults to `sqlite:///./storage/pdf_translation.db` so the cwd matters.
- **PyMuPDF SWIG warnings** — harmless, upstream library noise. Ignore.

---

## 10. Where to look next

- Full architecture and milestone plan: `/Users/applegarden/.claude/plans/ok-makes-plan-replicated-abelson.md`
- Project README (shorter quick-ref): [README.md](README.md)
- Original requirements: [CLAUDE.Md](CLAUDE.Md)

When you're ready to start M2, the entry point is the `extract_task` stub in [app/workers/tasks.py](app/workers/tasks.py) — it's where PyMuPDF parsing and OCR fallback will plug in.
