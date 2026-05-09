#!/usr/bin/env python3
"""Pilot a single book through Tesseract OCR to measure quality + speed.

Picks one PDF from ``storage/ref_books`` (or whatever path you pass), renders
every page to a high-DPI PNG, runs Tesseract with the Burmese language pack,
and prints per-page metrics + a sample of the extracted text so you can
eyeball the quality before committing to a full-corpus OCR run.

Usage::

    python scripts/ocr_pilot.py
    # or specific book:
    python scripts/ocr_pilot.py "storage/ref_books/မင်းသုဝဏ်/ကဗျာရေးချင်စာစပ်သင်.pdf"
    # or limit to first N pages:
    python scripts/ocr_pilot.py --max-pages 10
    # higher DPI (slower, sometimes better):
    python scripts/ocr_pilot.py --dpi 400

Reports:
  * Total pages processed and seconds elapsed
  * Time per page (so you can extrapolate to the full corpus)
  * Character count of recovered text
  * Tesseract average confidence score (per page)
  * Burmese-script-character ratio (sanity check that we got real Burmese)
  * A sample of the OCR'd text from the first non-empty page

Output text is written to ``storage/intermediate/ocr_pilot/<book>.txt`` so
you can compare side-by-side with the source PDF in a viewer.

This is a measurement script — it does NOT write to the database.
"""
from __future__ import annotations

import argparse
import logging
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pymupdf  # noqa: E402

log = logging.getLogger("ocr_pilot")

_DEFAULT_BOOK = (
    "storage/ref_books/မင်းသုဝဏ်/ကဗျာရေးချင်စာစပ်သင်.pdf"
)
_DEFAULT_DPI = 300

_BURMESE_RE = re.compile(r"[က-႟]")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)

    pdf_path = Path(args.book) if args.book else Path(_DEFAULT_BOOK)
    if not pdf_path.is_absolute():
        pdf_path = _REPO_ROOT / pdf_path
    if not pdf_path.exists():
        log.error("PDF not found: %s", pdf_path)
        return 2

    out_dir = _REPO_ROOT / "storage" / "intermediate" / "ocr_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / f"{pdf_path.stem}.txt"

    log.info("PILOT: %s", pdf_path)
    log.info("  DPI: %d, max pages: %s", args.dpi, args.max_pages or "all")
    log.info("  Output text: %s", out_txt)

    doc = pymupdf.open(pdf_path)
    total_pages = doc.page_count
    pages_to_process = (
        min(total_pages, args.max_pages) if args.max_pages else total_pages
    )
    log.info("  Total pages in book: %d, processing: %d", total_pages, pages_to_process)

    per_page_seconds: list[float] = []
    confidences: list[float] = []
    full_text_parts: list[str] = []
    burmese_chars = 0
    total_chars = 0

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i in range(pages_to_process):
            page = doc.load_page(i)
            t_page = time.monotonic()
            # Render to image at requested DPI.
            pix = page.get_pixmap(dpi=args.dpi, alpha=False)
            img_path = tmp_dir / f"p{i:04d}.png"
            pix.save(img_path)

            # Run tesseract — Burmese pack with default page-segmentation.
            text, mean_conf = _run_tesseract(img_path)
            elapsed = time.monotonic() - t_page

            per_page_seconds.append(elapsed)
            if mean_conf is not None:
                confidences.append(mean_conf)
            full_text_parts.append(f"--- PAGE {i + 1} ---\n{text}")
            burmese_chars += len(_BURMESE_RE.findall(text))
            total_chars += len(text)
            log.info(
                "  page %d/%d: %.2fs, conf=%s, chars=%d",
                i + 1,
                pages_to_process,
                elapsed,
                f"{mean_conf:.0f}" if mean_conf is not None else "?",
                len(text),
            )
    doc.close()
    total_elapsed = time.monotonic() - t0

    out_txt.write_text("\n\n".join(full_text_parts), encoding="utf-8")

    # ---------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------
    log.info("=" * 60)
    log.info("PILOT RESULTS")
    log.info("=" * 60)
    log.info("Book: %s", pdf_path.name)
    log.info("Pages processed: %d", pages_to_process)
    log.info("Wall-clock elapsed: %.1fs (%.2f sec/page)", total_elapsed, total_elapsed / pages_to_process)
    if per_page_seconds:
        log.info(
            "Per-page time: min=%.2fs, median=%.2fs, max=%.2fs",
            min(per_page_seconds),
            statistics.median(per_page_seconds),
            max(per_page_seconds),
        )
    if confidences:
        log.info(
            "Tesseract confidence: mean=%.0f, median=%.0f, min=%.0f",
            statistics.mean(confidences),
            statistics.median(confidences),
            min(confidences),
        )
    log.info("Total chars recovered: %d", total_chars)
    if total_chars > 0:
        log.info(
            "Burmese-script ratio: %.0f%% (%d Burmese / %d total)",
            100 * burmese_chars / total_chars,
            burmese_chars,
            total_chars,
        )
    log.info("Output saved to: %s", out_txt)
    log.info("")
    log.info("=" * 60)
    log.info("EXTRAPOLATION TO FULL CORPUS")
    log.info("=" * 60)
    sec_per_page = total_elapsed / pages_to_process
    pages_in_corpus = 26000  # rough estimate
    workers = 4
    log.info(
        "Single-threaded: ~%.1f hours for %d pages",
        pages_in_corpus * sec_per_page / 3600,
        pages_in_corpus,
    )
    log.info(
        "With %d parallel workers: ~%.1f hours",
        workers,
        pages_in_corpus * sec_per_page / 3600 / workers,
    )
    log.info("")
    log.info("=" * 60)
    log.info("SAMPLE OUTPUT (first non-empty page)")
    log.info("=" * 60)
    for part in full_text_parts:
        body = part.split("---\n", 1)[-1].strip()
        if body:
            log.info("%s", body[:1500])
            break

    return 0


def _run_tesseract(img_path: Path) -> tuple[str, float | None]:
    """Run tesseract on an image, return (text, mean_confidence).

    We call tesseract twice:
      1. Plain text mode for the actual output
      2. tsv mode for the per-word confidence scores

    Calling twice doubles per-page time but is the cleanest way to get both
    plain text and confidence — tesseract's tsv-only mode produces tab-
    separated output that's annoying to reassemble into prose. Acceptable
    for a pilot; for production OCR we'd use tesserocr (Python bindings)
    and grab both in one inference.
    """
    text = subprocess.run(
        ["tesseract", str(img_path), "stdout", "-l", "mya"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout

    tsv = subprocess.run(
        ["tesseract", str(img_path), "stdout", "-l", "mya", "tsv"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout

    confidences: list[float] = []
    for line in tsv.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 12:
            continue
        try:
            conf = float(cols[10])
        except ValueError:
            continue
        if conf >= 0:
            confidences.append(conf)
    mean_conf = statistics.mean(confidences) if confidences else None
    return text, mean_conf


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "book",
        nargs="?",
        default=None,
        help="Path to a single PDF to pilot. Defaults to a small Min Thu Wun book.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=_DEFAULT_DPI,
        help="Render DPI before OCR. 300 is the Tesseract sweet spot; 400 sometimes helps on small fonts.",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit to first N pages (faster pilot). Default: process all pages.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
