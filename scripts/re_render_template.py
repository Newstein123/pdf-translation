#!/usr/bin/env python3
"""Re-render an already-translated job's PDF using the TEMPLATE export path.

Useful for retroactively producing a clean Burmese book PDF from a job
that finished under the legacy PRESERVE rebuild — no Gemini calls, no
re-translation. Pulls translated chunks from the DB and pipes them through
``pdf_template.render_template_pdf``.

Usage::

    python scripts/re_render_template.py <job_id>
    python scripts/re_render_template.py <job_id> --title "The Time Machine" --title-target "အချိန်စက်"
    python scripts/re_render_template.py <job_id> --author "H. G. Wells"

Without --title / --author flags, falls back to (in order):
  1. Job columns ``title_target`` / ``title_source`` / ``author_*``
  2. Auto-detection from the source PDF's metadata
  3. Filename for title

Writes the new PDF alongside the existing one with suffix ``.my.template.pdf``
so the original is preserved and you can compare side-by-side.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.db.models import Job, TranslationChunk  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services import layout_detection, pdf_template  # noqa: E402

log = logging.getLogger("re_render_template")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    settings = get_settings()

    with SessionLocal() as db:
        job = db.get(Job, args.job_id)
        if job is None:
            log.error("job %s not found", args.job_id)
            return 2

        log.info("=== job: %s ===", job.id)
        log.info("  filename: %s", job.original_filename)
        log.info("  status:   %s", job.status.value)
        log.info("  pages:    %s", job.page_count)

        # Title/author resolution.
        title_source = args.title or job.title_source
        title_target = args.title_target or job.title_target
        author_source = args.author or job.author_source
        author_target = args.author_target or job.author_target

        if not title_source or not author_source:
            try:
                detected = layout_detection.detect_book_metadata(Path(job.file_path))
                title_source = title_source or detected.title
                author_source = author_source or detected.author
                log.info(
                    "  auto-detected title=%r, author=%r",
                    detected.title,
                    detected.author,
                )
            except Exception:
                log.exception("metadata detection failed; proceeding without")

        log.info("  title_source: %s", title_source)
        log.info("  title_target: %s", title_target)
        log.info("  author_source: %s", author_source)
        log.info("  author_target: %s", author_target)

        # Persist auto-detected metadata back to the job so subsequent
        # re-renders don't re-detect. Source-side only — target stays empty
        # unless the operator passed --title-target.
        dirty = False
        if title_source and not job.title_source:
            job.title_source = title_source
            dirty = True
        if title_target and not job.title_target:
            job.title_target = title_target
            dirty = True
        if author_source and not job.author_source:
            job.author_source = author_source
            dirty = True
        if author_target and not job.author_target:
            job.author_target = author_target
            dirty = True
        if dirty:
            db.commit()

        # Pull all translated blocks.
        chunks = (
            db.query(TranslationChunk)
            .filter(TranslationChunk.job_id == job.id)
            .order_by(TranslationChunk.chunk_index)
            .all()
        )
        translations_by_block: dict[str, str] = {}
        for c in chunks:
            for entry in c.translated_blocks_json or []:
                bid = entry.get("block_id")
                translated = entry.get("translated") or ""
                if bid and translated:
                    translations_by_block[bid] = translated

        log.info("  translated blocks: %d", len(translations_by_block))
        if not translations_by_block:
            log.error("  no translated blocks — nothing to render")
            return 2

    structure_path = settings.intermediate_dir / args.job_id / "structure.json"
    if not structure_path.exists():
        log.error("structure.json missing at %s", structure_path)
        return 2

    output_dir = settings.outputs_dir / args.job_id
    base = Path(job.original_filename).stem or "translated"
    output_name = f"{base}.my.template.pdf"

    config = pdf_template.TemplateConfig.from_settings(
        overrides=job.export_template_overrides_json,
        title_source=title_source,
        title_target=title_target,
        author_source=author_source,
        author_target=author_target,
    )
    log.info("  template config: %s", config)

    log.info("rendering...")
    result = pdf_template.render_template_pdf(
        structure_path=structure_path,
        translations_by_block=translations_by_block,
        output_dir=output_dir,
        output_name=output_name,
        config=config,
    )
    log.info("done.")
    log.info("  output:    %s", result.output_path)
    log.info("  pages:     %d", result.pages_written)
    log.info("  chapters:  %d", result.chapters_rendered)
    log.info("  blocks:    %d", result.blocks_placed)
    log.info("  size:      %d bytes", result.output_path.stat().st_size)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_id")
    p.add_argument("--title", default=None, help="Override source title.")
    p.add_argument(
        "--title-target", default=None, help="Burmese-language title (running header)."
    )
    p.add_argument("--author", default=None, help="Override source author.")
    p.add_argument(
        "--author-target", default=None, help="Burmese-language author."
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
