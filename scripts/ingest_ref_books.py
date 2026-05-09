#!/usr/bin/env python3
"""Ingest reference books from ``storage/ref_books`` into the database.

Reads PDFs from per-translator subdirectories (one folder per translator),
extracts clean Burmese text, slices it into passage-sized chunks, classifies
each chunk by mode (NARRATION / DIALOGUE / POETRY / EXPOSITION) using fast
deterministic heuristics, then writes them to the ``reference_passages``
table.

After ingest, optionally generates a per-translator ``voice_card``: a
distilled style brief that gets injected into the cached prompt prefix on
each translation job — voice anchoring without paying retrieval tokens
per chunk.

Usage::

    # Activate venv first.
    python scripts/ingest_ref_books.py
    # Or limit to one translator:
    python scripts/ingest_ref_books.py --translator မင်းသုဝဏ်
    # Or skip voice-card generation (Gemini calls):
    python scripts/ingest_ref_books.py --skip-voice-cards
    # Or dry-run (extract + classify, don't write):
    python scripts/ingest_ref_books.py --dry-run

Cost: extraction is local (free). Voice card generation is one Gemini Flash
call per translator (~$0.01 per card). Embeddings are skipped by default
because the corpus is target-only — retrieval relies on translator + mode
filters, not embedding similarity. Pass ``--with-embeddings`` to compute
them anyway (one call per passage; ~$0.10 for the full 100-book corpus).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Make ``app.*`` importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import pymupdf  # noqa: E402  (after sys.path setup)
from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.models import ReferenceMode, ReferencePassage  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services import voice_card  # noqa: E402
from app.utils.tokens import count_tokens  # noqa: E402

log = logging.getLogger("ingest_ref_books")

# Passage targets. ~600 tokens is large enough to capture sentence-rhythm
# patterns but small enough that retrieval can pick a focused exemplar.
_PASSAGE_TARGET_TOKENS = 600
_PASSAGE_MIN_TOKENS = 150
_PASSAGE_MAX_TOKENS = 1000

# Author → work-type map. ORIGINAL = the author writes in Burmese natively
# (the corpus shows authentic, non-translated Burmese voice). TRANSLATION =
# the author is a translator (the corpus shows their voice when rendering
# foreign source into Burmese). Voice cards are framed differently for the
# two cases — original-Burmese authors teach native voice patterns;
# translators teach how to negotiate foreign source into Burmese.
#
# Folder names use Burmese script verbatim; if you add a new author folder,
# add an entry here. Default is TRANSLATION because the majority case in
# this corpus is published Burmese translations of foreign works.
_AUTHOR_WORK_TYPE: dict[str, str] = {
    "မင်းသုဝဏ်": "ORIGINAL",      # Min Thu Wun — original Burmese poet/author
    "ShweOoDaung": "TRANSLATION",
    "ဗန်းမော်တင်အောင်": "TRANSLATION",  # Banmaw Tin Aung
    "မြသန်းတင့်": "TRANSLATION",   # Mya Than Tint
    "ဖေမြင့်": "TRANSLATION",      # Hpe Myint — translates Western self-help / personal-development titles
}

# Burmese script + dialogue / poetry detection.
_BURMESE_RE = re.compile(r"[က-႟]")
# Burmese-specific dialogue markers:
#   - "ဟု" / "ဟူ၍"  — classical "said" particles
#   - Smart quotes / ASCII double-quotes
# We deliberately do NOT match "၏" or "၊" because those are generic
# sentence/clause separators that appear in plain narration; including
# them earlier mis-classified ~75% of prose as DIALOGUE.
_DIALOGUE_RE = re.compile(r"ဟု|ဟူ၍|“|”|„|‟|\"")
# Poetry detection: many short lines in a row.
_LINE_RE = re.compile(r"\n+")


@dataclass
class _Passage:
    text: str
    token_count: int
    mode: ReferenceMode


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)

    settings = get_settings()
    root = settings.storage_root / "ref_books"
    if not root.exists():
        log.error("ref books root does not exist: %s", root)
        return 2

    translator_dirs = _list_translator_dirs(root, only=args.translator)
    if not translator_dirs:
        log.error("no translator folders found under %s", root)
        return 2

    log.info("ingesting %d translator(s) from %s", len(translator_dirs), root)
    total_passages = 0
    per_translator_counts: dict[str, int] = {}

    with SessionLocal() as session:
        for tdir in translator_dirs:
            translator = tdir.name
            log.info("--- translator: %s ---", translator)
            count = _ingest_translator(
                session,
                translator=translator,
                directory=tdir,
                dry_run=args.dry_run,
                with_embeddings=args.with_embeddings,
                replace=args.replace,
            )
            per_translator_counts[translator] = count
            total_passages += count

        if not args.dry_run:
            session.commit()
            log.info("committed %d new passages across %d translators",
                     total_passages, len(per_translator_counts))

        if args.skip_voice_cards or args.dry_run:
            log.info("voice card generation skipped (flag or dry-run)")
        else:
            for translator, count in per_translator_counts.items():
                if count == 0:
                    log.info("voice card for %s skipped (no new passages)", translator)
                    continue
                _safe_generate_voice_card(session, translator)
            session.commit()

    log.info("done. summary:")
    for t, c in per_translator_counts.items():
        log.info("  %s: %d passages", t, c)
    return 0


# ---------------------------------------------------------------------------
# Ingest per translator
# ---------------------------------------------------------------------------


def _ingest_translator(
    session: Session,
    *,
    translator: str,
    directory: Path,
    dry_run: bool,
    with_embeddings: bool,
    replace: bool,
) -> int:
    # Recurse so nested subfolders (e.g. an unzipped archive landing in
    # ``<author>/Mya Than Tint/``) get picked up alongside flat PDFs.
    # Dedupe on resolved path so symlinks / case-insensitive filesystems
    # don't double-count.
    seen: set[Path] = set()
    pdfs: list[Path] = []
    for p in sorted(directory.rglob("*.pdf")):
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        pdfs.append(p)
    if not pdfs:
        log.info("  no PDFs in %s", directory)
        return 0
    work_type = _AUTHOR_WORK_TYPE.get(translator, "TRANSLATION")
    log.info("  found %d PDF(s) [work_type=%s]", len(pdfs), work_type)

    if replace and not dry_run:
        # Soft-replace: deactivate prior entries so a re-ingest doesn't double
        # the corpus. We don't hard-delete so we can audit prior runs.
        rows = list(
            session.execute(
                select(ReferencePassage).where(
                    ReferencePassage.translator == translator,
                    ReferencePassage.active.is_(True),
                )
            ).scalars()
        )
        for row in rows:
            row.active = False
        if rows:
            log.info("  deactivated %d existing passages for replacement", len(rows))

    added = 0
    for pdf_path in pdfs:
        book_title = pdf_path.stem
        try:
            passages = _passages_from_pdf(pdf_path)
        except Exception:
            log.exception("  failed to extract %s", pdf_path)
            continue
        if not passages:
            log.info("  %s: no extractable passages", pdf_path.name)
            continue
        log.info("  %s: %d passages", pdf_path.name, len(passages))

        if dry_run:
            added += len(passages)
            continue

        for p in passages:
            embedding = None
            if with_embeddings:
                try:
                    from app.services.reference_library import _compute_embedding

                    embedding = _compute_embedding(p.text)
                except Exception:
                    log.exception("    embedding failed for a passage; storing without")
                    embedding = None
            session.add(
                ReferencePassage(
                    source_text="",
                    target_text=p.text,
                    source_lang="en",  # source is unknown but field is non-null
                    target_lang="my",
                    mode=p.mode,
                    book_title=book_title,
                    translator=translator,
                    embedding_json=embedding,
                    embedding_model=("text-embedding-004" if embedding else None),
                    active=True,
                )
            )
            added += 1
        # Flush per book so a later failure doesn't lose the books that
        # already extracted cleanly.
        session.flush()

    return added


# ---------------------------------------------------------------------------
# PDF → passages
# ---------------------------------------------------------------------------


def _passages_from_pdf(pdf_path: Path) -> list[_Passage]:
    """Open a PDF, pull text per page, slice into passage-sized chunks,
    classify each by mode.

    Skips books that aren't readable as plain Unicode Burmese:
      * Scanned image PDFs (no text layer) — would need OCR.
      * Legacy Latin-1 fonts (text in Win-Burmese / Win-Resarcher) — would
        need a font-mapping decoder.

    Returns an empty list with a logged ``[encoding=...]`` note so the
    caller can surface the skip reason. The voice-card pipeline simply
    sees an empty passage list and emits no card for that book."""
    out: list[_Passage] = []
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        log.exception("    pymupdf open failed: %s", pdf_path)
        return []
    try:
        encoding = _detect_encoding(doc)
        if encoding != "unicode":
            log.info("    %s [encoding=%s] — skipping (needs OCR or decoder)",
                     pdf_path.name, encoding)
            return []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            text = _normalize_whitespace(text)
            if not text.strip():
                continue
            if not _BURMESE_RE.search(text):
                continue
            for chunk in _slice_into_passages(text):
                mode = _classify_mode(chunk)
                out.append(
                    _Passage(
                        text=chunk,
                        token_count=count_tokens(chunk),
                        mode=mode,
                    )
                )
    finally:
        doc.close()
    return out


def _detect_encoding(doc) -> str:
    """Classify a PDF's text encoding by sampling 3 pages.

    Returns one of ``"unicode"`` (clean Unicode Burmese, ingestible),
    ``"legacy_latin1"`` (old Win-Burmese fonts), ``"scanned"`` (no text
    layer), or ``"unknown"`` (text but no recognizable Burmese script).
    """
    n = doc.page_count
    if n == 0:
        return "scanned"
    sample_indices = sorted({n // 4, n // 2, 3 * n // 4, 0})[:4]
    sample = "".join(doc.load_page(min(i, n - 1)).get_text("text") for i in sample_indices)
    if not sample.strip():
        return "scanned"

    unicode_burmese = len(_BURMESE_RE.findall(sample))
    latin1_extended = sum(1 for ch in sample if 160 < ord(ch) < 256)
    ascii_chars = sum(1 for ch in sample if 32 < ord(ch) < 127)

    if unicode_burmese > 100:
        return "unicode"
    if latin1_extended > 200:
        return "legacy_latin1"
    if ascii_chars > 200:
        return "unknown"
    return "scanned"


def _slice_into_passages(text: str) -> Iterable[str]:
    """Greedy passage slicer.

    Walk paragraphs; accumulate until target tokens reached; emit and reset.
    Skips paragraphs below the minimum so we don't ship 30-token noise rows.
    Splits over-long single paragraphs by sentence end so we never blow the
    max-tokens cap.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    buf: list[str] = []
    buf_tokens = 0
    for para in paragraphs:
        para_tokens = count_tokens(para)
        if para_tokens > _PASSAGE_MAX_TOKENS:
            # Single paragraph too big; split on Burmese sentence ends.
            for piece in _split_by_sentence(para, _PASSAGE_MAX_TOKENS):
                yield piece
            continue
        if buf_tokens + para_tokens > _PASSAGE_TARGET_TOKENS and buf:
            joined = "\n\n".join(buf)
            if count_tokens(joined) >= _PASSAGE_MIN_TOKENS:
                yield joined
            buf = [para]
            buf_tokens = para_tokens
        else:
            buf.append(para)
            buf_tokens += para_tokens
    if buf:
        joined = "\n\n".join(buf)
        if count_tokens(joined) >= _PASSAGE_MIN_TOKENS:
            yield joined


def _split_by_sentence(text: str, max_tokens: int) -> Iterable[str]:
    sentences = re.split(r"(?<=[။၏!?])\s+", text)
    buf: list[str] = []
    buf_tokens = 0
    for sent in sentences:
        st = sent.strip()
        if not st:
            continue
        t = count_tokens(st)
        if buf_tokens + t > max_tokens and buf:
            yield " ".join(buf)
            buf = [st]
            buf_tokens = t
        else:
            buf.append(st)
            buf_tokens += t
    if buf:
        yield " ".join(buf)


def _classify_mode(text: str) -> ReferenceMode:
    """Cheap deterministic classifier.

    Heuristics in priority order:
      1. POETRY — short lines AND very few sentence enders. PDF column
         wrap also produces short lines, but PROSE keeps its sentence
         enders inside the wrapped lines, so the absence of ။ / ၏
         distinguishes real poetry from line-wrapped prose.
      2. DIALOGUE — strong dialogue markers (ဟု, smart quotes) appear at
         normal density.
      3. EXPOSITION — high digit / numbered density.
      4. NARRATION — default for everything else.
    """
    if not text:
        return ReferenceMode.OTHER

    lines = [line for line in text.split("\n") if line.strip()]
    if lines:
        avg_line_len = sum(len(line) for line in lines) / len(lines)
        sentence_enders = len(re.findall(r"[။၏]", text))
        # Sentence enders per 1000 chars. Prose runs ~5–15; poetry ~0–3.
        ender_rate = sentence_enders / max(1, len(text)) * 1000
        if avg_line_len < 40 and ender_rate < 4 and len(lines) >= 4:
            return ReferenceMode.POETRY

    dialogue_hits = len(_DIALOGUE_RE.findall(text))
    if dialogue_hits >= 3 or (dialogue_hits >= 1 and len(text) < 1500):
        return ReferenceMode.DIALOGUE

    digits = len(re.findall(r"[0-9၀-၉]", text))
    if digits / max(1, len(text)) > 0.04:
        return ReferenceMode.EXPOSITION

    return ReferenceMode.NARRATION


def _normalize_whitespace(text: str) -> str:
    # Collapse runs of spaces but preserve paragraph breaks.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Voice card generation
# ---------------------------------------------------------------------------


def _safe_generate_voice_card(session: Session, translator: str) -> None:
    work_type = _AUTHOR_WORK_TYPE.get(translator)
    log.info(
        "generating voice card for %s [work_type=%s] ...",
        translator,
        work_type or "(unset, default TRANSLATION)",
    )
    try:
        card = voice_card.generate_voice_card(
            session, translator, force=True, work_type=work_type
        )
        log.info(
            "  card built: passages=%d, tokens≈%d",
            card.source_passage_count,
            count_tokens(card.card_text),
        )
    except voice_card.VoiceCardError as e:
        log.warning("  voice card failed for %s: %s", translator, e)
    except Exception:
        log.exception("  unexpected error generating voice card for %s", translator)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _list_translator_dirs(root: Path, *, only: str | None) -> list[Path]:
    out: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        if only and entry.name != only:
            continue
        out.append(entry)
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--translator",
        default=None,
        help="Limit ingest to this translator's folder (defaults to all).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract + classify but don't write to DB.",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Deactivate prior passages for each translator before ingest.",
    )
    p.add_argument(
        "--with-embeddings",
        action="store_true",
        help=(
            "Compute embeddings for each passage. Skipped by default — "
            "voice cards + translator/mode filters cover most use cases."
        ),
    )
    p.add_argument(
        "--skip-voice-cards",
        action="store_true",
        help="Skip the per-translator voice-card generation step.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
