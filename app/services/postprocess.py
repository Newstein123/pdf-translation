"""Post-processing: glossary validation across all completed chunks.

For now this is a thin wrapper that scans each completed chunk's translated
text against the job's glossary and flags any chunks that need a retry.
Name-consistency validation will land alongside the M3 retry loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.db.models import GlossaryEntry, TranslationChunk
from app.services.glossary import GlossaryViolation, find_violations

log = logging.getLogger(__name__)


@dataclass
class ChunkValidation:
    chunk_index: int
    violations: list[GlossaryViolation]


def validate_chunks(
    chunks: list[TranslationChunk],
    glossary: list[GlossaryEntry],
) -> list[ChunkValidation]:
    """Return per-chunk glossary violations. Empty if everything passes."""
    if not glossary:
        return []
    out: list[ChunkValidation] = []
    for c in chunks:
        if not c.translated_text:
            continue
        v = find_violations(
            source_text=c.source_text,
            translated_text=c.translated_text,
            glossary=glossary,
        )
        if v:
            out.append(ChunkValidation(chunk_index=c.chunk_index, violations=v))
    return out
