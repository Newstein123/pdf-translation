"""End-to-end tests for the rebuild_task: translate → PDF + EPUB outputs."""

from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pymupdf

from app.db.models import Category, Job, JobStatus, Output, OutputFormat
from app.db.session import SessionLocal
from app.services.translation import ChunkTranslationResult, DiscoveredName, TranslatedBlock


_TEXT = (
    "Alice walked through the Forest of Eldar to find Master Bell. "
    "The journey was long, and the night was cold. " * 8
)


def _make_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    # Use insert_textbox so the resulting block has a multi-line bbox tall
    # enough to receive a translation. insert_text produces a single-line bbox
    # which is too short for replacement text and not representative of real PDFs.
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 720), _TEXT, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def _seed_translated(pdf_path: Path) -> str:
    """Drive a job through extract → classify → translate (Gemini stubbed)."""
    with SessionLocal() as db:
        job = Job(
            original_filename=pdf_path.name,
            file_path=str(pdf_path),
            file_size_bytes=pdf_path.stat().st_size,
            status=JobStatus.UPLOADED,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    from app.workers import tasks

    with patch("app.workers.tasks.classify_task.delay"):
        tasks.extract_task.run(job_id)
    tasks.classify_task.run(job_id)
    with SessionLocal() as db:
        db.get(Job, job_id).category = Category.NOVEL
        db.commit()

    def stub(*, blocks, **_kw):
        return ChunkTranslationResult(
            blocks=[
                TranslatedBlock(block_id=b["block_id"], translated="ဘာသာပြန်: " + b["source"][:30])
                for b in blocks
            ],
            new_names=[DiscoveredName(source="Alice", burmese="အဲလစ်")],
            summary_for_next_chunk="prev summary",
        )

    with (
        patch.object(tasks.translation, "translate_chunk", side_effect=stub),
        patch("app.workers.tasks.rebuild_task.delay"),
    ):
        tasks.translate_task.run(job_id)
    return job_id


def test_rebuild_task_produces_pdf_and_epub(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_translated(pdf)

    from app.workers import tasks

    tasks.rebuild_task.run(job_id)

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        outputs = db.query(Output).filter(Output.job_id == job_id).all()

    assert job.status == JobStatus.COMPLETED
    formats = {o.format for o in outputs}
    assert OutputFormat.PDF in formats
    assert OutputFormat.EPUB in formats

    pdf_path = Path(next(o.file_path for o in outputs if o.format == OutputFormat.PDF))
    epub_path = Path(next(o.file_path for o in outputs if o.format == OutputFormat.EPUB))
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    assert epub_path.exists() and epub_path.stat().st_size > 0

    # Output PDF must open and contain the translation marker. PyMuPDF inserts
    # newlines at visual line boundaries when extracting text, so collapse
    # whitespace before checking inclusion.
    out_doc = pymupdf.open(pdf_path)
    try:
        full = "\n".join(out_doc.load_page(i).get_text() for i in range(out_doc.page_count))
    finally:
        out_doc.close()
    flat = "".join(full.split())
    assert "ဘာသာပြန်" in flat, f"marker not found in extracted text:\n{full[:500]}"

    # EPUB must be a valid zip containing at least one chapter XHTML.
    with ZipFile(epub_path) as zf:
        names = zf.namelist()
        chapters = [n for n in names if n.endswith(".xhtml") and n != "nav.xhtml"]
        assert chapters, f"no chapter files in EPUB: {names}"
        body = zf.read(chapters[0]).decode("utf-8")
        assert "ဘာသာပြန်" in body


def test_outputs_endpoints(client, auth_headers, tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "src.pdf")
    job_id = _seed_translated(pdf)

    from app.workers import tasks

    tasks.rebuild_task.run(job_id)

    r = client.get(f"/api/jobs/{job_id}/outputs", headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    formats = {o["format"] for o in body["outputs"]}
    assert formats == {"PDF", "EPUB"}

    # Download PDF
    r_pdf = client.get(f"/api/jobs/{job_id}/outputs/pdf", headers=auth_headers)
    assert r_pdf.status_code == 200
    assert r_pdf.headers["content-type"] == "application/pdf"
    assert r_pdf.content[:4] == b"%PDF"

    # Download EPUB
    r_epub = client.get(f"/api/jobs/{job_id}/outputs/EPUB", headers=auth_headers)
    assert r_epub.status_code == 200
    assert r_epub.headers["content-type"] == "application/epub+zip"
    assert r_epub.content[:2] == b"PK"  # zip magic

    # Bad format → 400
    r_bad = client.get(f"/api/jobs/{job_id}/outputs/docx", headers=auth_headers)
    assert r_bad.status_code == 400
