from pathlib import Path
from unittest.mock import patch


def test_health(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_requires_auth(client, sample_pdf: Path) -> None:
    with sample_pdf.open("rb") as fh:
        r = client.post("/api/jobs", files={"file": ("sample.pdf", fh, "application/pdf")})
    assert r.status_code in (401, 403)


def test_upload_and_fetch_status(client, auth_headers: dict, sample_pdf: Path) -> None:
    with patch("app.workers.tasks.extract_task.delay") as mock_delay:
        with sample_pdf.open("rb") as fh:
            r = client.post(
                "/api/jobs",
                headers=auth_headers,
                files={"file": ("sample.pdf", fh, "application/pdf")},
            )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "UPLOADED"
    assert body["original_filename"] == "sample.pdf"
    job_id = body["id"]
    mock_delay.assert_called_once_with(job_id)

    r2 = client.get(f"/api/jobs/{job_id}", headers=auth_headers)
    assert r2.status_code == 200
    assert r2.json()["page_count"] == 1


def test_upload_rejects_non_pdf(client, auth_headers: dict, tmp_path: Path) -> None:
    f = tmp_path / "note.txt"
    f.write_text("not a pdf")
    with f.open("rb") as fh:
        r = client.post(
            "/api/jobs",
            headers=auth_headers,
            files={"file": ("note.txt", fh, "text/plain")},
        )
    assert r.status_code == 400


def test_get_unknown_job_404s(client, auth_headers: dict) -> None:
    r = client.get("/api/jobs/00000000-0000-0000-0000-000000000000", headers=auth_headers)
    assert r.status_code == 404


def test_list_jobs_returns_recent_first(client, auth_headers: dict, sample_pdf) -> None:
    """GET /api/jobs returns recent jobs newest-first, used by the UI history panel."""
    from unittest.mock import patch

    created_ids = []
    with patch("app.workers.tasks.extract_task.delay"):
        for _ in range(3):
            with sample_pdf.open("rb") as fh:
                r = client.post(
                    "/api/jobs",
                    headers=auth_headers,
                    files={"file": ("sample.pdf", fh, "application/pdf")},
                )
            assert r.status_code == 201
            created_ids.append(r.json()["id"])

    r = client.get("/api/jobs", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    listed_ids = [j["id"] for j in body]
    for cid in created_ids:
        assert cid in listed_ids


def test_list_jobs_respects_limit(client, auth_headers: dict, sample_pdf) -> None:
    r = client.get("/api/jobs?limit=1", headers=auth_headers)
    assert r.status_code == 200
    assert len(r.json()) <= 1


def test_list_jobs_requires_auth(client) -> None:
    r = client.get("/api/jobs")
    assert r.status_code in (401, 403)
