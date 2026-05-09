"""Build translation chunks from a normalized structure.json.

Rules:
- Never split inside a single block (paragraph/heading/ocr); blocks are atomic.
- Aim for ``CHUNK_TARGET_TOKENS``; never exceed ``CHUNK_HARD_CAP_TOKENS``.
- If a single block exceeds the hard cap, it becomes its own oversized chunk
  (we'd rather translate one large block than corrupt block boundaries).
- Educational documents use a smaller target for tighter terminology control
  via ``EDUCATIONAL_TARGET_RATIO``.

Output: list of ``ChunkSpec`` ready to persist as ``TranslationChunk`` rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import get_settings
from app.db.models import Category
from app.utils.tokens import count_tokens

EDUCATIONAL_TARGET_RATIO = 0.7  # smaller chunks for terminology-heavy docs


@dataclass
class ChunkSpec:
    chunk_index: int
    page_range_start: int
    page_range_end: int
    blocks: list[dict] = field(default_factory=list)  # [{"block_id": str, "source": str}]
    source_text: str = ""
    token_count: int = 0


def build_chunks(
    structure_path: Path,
    *,
    category: Category,
    target_tokens_override: int | None = None,
    hard_cap_tokens_override: int | None = None,
    skip_pages: set[int] | None = None,
) -> list[ChunkSpec]:
    """Build chunks. Per-job overrides win over global settings; the
    EDUCATIONAL ratio still applies when no explicit override is set so the
    historical defaults are preserved.

    ``skip_pages`` (M15) is a set of 1-indexed page numbers to exclude
    from chunking — translation never sees those pages, saving Gemini
    tokens. The same set is applied at rebuild time to drop the pages
    from the output PDF.
    """
    s = get_settings()
    target = target_tokens_override if target_tokens_override else s.chunk_target_tokens
    hard_cap = (
        hard_cap_tokens_override if hard_cap_tokens_override else s.chunk_hard_cap_tokens
    )
    # Apply the educational scaling only when the user didn't set an explicit
    # override — explicit user intent should take precedence.
    if category == Category.EDUCATIONAL and not target_tokens_override:
        target = int(target * EDUCATIONAL_TARGET_RATIO)

    skip = skip_pages or set()

    with structure_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    chunks: list[ChunkSpec] = []
    current = ChunkSpec(chunk_index=0, page_range_start=0, page_range_end=0)

    def flush() -> None:
        nonlocal current
        if not current.blocks:
            return
        current.source_text = "\n\n".join(b["source"] for b in current.blocks)
        chunks.append(current)
        next_index = current.chunk_index + 1
        current = ChunkSpec(chunk_index=next_index, page_range_start=0, page_range_end=0)

    for page in data.get("pages", []):
        page_no = int(page.get("page_number", 0))
        # M15: skip user-excluded pages entirely. Their blocks never enter a
        # chunk so Gemini never sees them.
        if page_no in skip:
            continue
        for block in page.get("blocks", []):
            text = (block.get("text") or "").strip()
            if not text:
                continue
            block_id = block.get("block_id") or f"p{page_no}-{len(current.blocks)}"
            tokens = count_tokens(text)

            # Block alone exceeds hard cap → flush current, then emit oversized chunk solo.
            if tokens > hard_cap:
                flush()
                solo = ChunkSpec(
                    chunk_index=current.chunk_index,
                    page_range_start=page_no,
                    page_range_end=page_no,
                    blocks=[{"block_id": block_id, "source": text}],
                    source_text=text,
                    token_count=tokens,
                )
                chunks.append(solo)
                current = ChunkSpec(
                    chunk_index=solo.chunk_index + 1, page_range_start=0, page_range_end=0
                )
                continue

            # Adding this block would exceed hard cap → flush first.
            if current.blocks and current.token_count + tokens > hard_cap:
                flush()

            if not current.blocks:
                current.page_range_start = page_no
            current.page_range_end = page_no
            current.blocks.append({"block_id": block_id, "source": text})
            current.token_count += tokens

            if current.token_count >= target:
                flush()

    flush()
    return chunks
