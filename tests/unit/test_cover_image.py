"""Tests for cover-image upload + render (M16)."""
from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from app.services.cover_image import CoverImageError, save_cover_image


def _png_bytes(width: int = 400, height: int = 600, color=(200, 50, 50)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# save_cover_image
# ---------------------------------------------------------------------------


def test_save_cover_image_writes_jpeg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    path = save_cover_image(job_id="test-job", raw_bytes=_png_bytes())
    assert path.exists()
    assert path.suffix == ".jpg"
    # Round-trip the saved file to confirm it's a real JPEG and decodable.
    with Image.open(path) as im:
        im.load()
        assert im.format == "JPEG"


def test_save_cover_image_resizes_oversized(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    big = _png_bytes(width=4000, height=6000)
    path = save_cover_image(job_id="big", raw_bytes=big)
    with Image.open(path) as im:
        # Long edge should be capped by the service's max-long-edge constant.
        assert max(im.size) <= 1600


def test_save_cover_image_rejects_empty() -> None:
    with pytest.raises(CoverImageError, match="empty"):
        save_cover_image(job_id="x", raw_bytes=b"")


def test_save_cover_image_rejects_garbage() -> None:
    with pytest.raises(CoverImageError):
        save_cover_image(job_id="x", raw_bytes=b"not a real image, just bytes")


def test_save_cover_image_rejects_fully_transparent_png(
    tmp_path: Path, monkeypatch
) -> None:
    """RGBA PNG with alpha=0 throughout → flattens to all-white. Reject
    with a user-readable error so users don't get a silent blank cover."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()

    img = Image.new("RGBA", (400, 600), (255, 0, 0, 0))  # red but alpha=0
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    with pytest.raises(CoverImageError, match="blank|transparent"):
        save_cover_image(job_id="alpha-zero", raw_bytes=buf.getvalue())


def test_save_cover_image_rejects_solid_white_jpeg(
    tmp_path: Path, monkeypatch
) -> None:
    """JPG that happens to be solid white — same defensive check."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()

    img = Image.new("RGB", (400, 600), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    with pytest.raises(CoverImageError, match="blank|transparent"):
        save_cover_image(job_id="solid-white", raw_bytes=buf.getvalue())


def test_save_cover_image_accepts_image_with_actual_content(
    tmp_path: Path, monkeypatch
) -> None:
    """Sanity: a real photo-like image passes the uniform-white guard."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()

    # A gradient — definitely not uniform white.
    img = Image.new("RGB", (400, 600))
    for y in range(img.height):
        for x in range(img.width):
            img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    path = save_cover_image(job_id="gradient", raw_bytes=buf.getvalue())
    assert path.exists()


# ---------------------------------------------------------------------------
# Cover renders into TEMPLATE export PDF
# ---------------------------------------------------------------------------


def test_template_render_includes_cover_when_provided(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()

    from app.services.pdf_template import TemplateConfig, render_template_pdf

    # Save a cover image.
    cover_path = save_cover_image(job_id="t", raw_bytes=_png_bytes(800, 1200))

    # Build a minimal structure + translations.
    import json

    structure_path = tmp_path / "structure.json"
    structure_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {"total_pages": 1, "total_blocks": 1, "scanned_pages": []},
                "pages": [
                    {
                        "page_number": 1,
                        "width": 400,
                        "height": 500,
                        "is_scanned": False,
                        "blocks": [
                            {"block_id": "b0", "type": "paragraph", "text": "hello"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cfg = TemplateConfig.from_settings()
    result = render_template_pdf(
        structure_path=structure_path,
        translations_by_block={"b0": "မင်္ဂလာပါ"},
        output_dir=tmp_path / "out",
        output_name="cov.pdf",
        config=cfg,
        cover_image_path=cover_path,
    )
    assert result.output_path.exists()
    doc = pymupdf.open(result.output_path)
    try:
        # Page 1 should be the cover (i.e. contain the embedded image we
        # just saved). PyMuPDF exposes images on a page via get_images().
        page0_images = doc.load_page(0).get_images()
        assert page0_images, "no image embedded on page 1 — cover not rendered"
    finally:
        doc.close()


def test_template_render_no_cover_when_path_omitted(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()

    import json

    from app.services.pdf_template import TemplateConfig, render_template_pdf

    structure_path = tmp_path / "structure.json"
    structure_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {"total_pages": 1, "total_blocks": 1, "scanned_pages": []},
                "pages": [
                    {
                        "page_number": 1,
                        "width": 400,
                        "height": 500,
                        "is_scanned": False,
                        "blocks": [
                            {"block_id": "b0", "type": "paragraph", "text": "hi"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = TemplateConfig.from_settings()
    result = render_template_pdf(
        structure_path=structure_path,
        translations_by_block={"b0": "ဟိုင်း"},
        output_dir=tmp_path / "out",
        output_name="nocov.pdf",
        config=cfg,
        # cover_image_path omitted
    )
    doc = pymupdf.open(result.output_path)
    try:
        # Page 0 should NOT have a raster image embedded (no cover).
        # PyMuPDF Story embeds fonts / not images for plain text.
        assert not doc.load_page(0).get_images()
    finally:
        doc.close()
