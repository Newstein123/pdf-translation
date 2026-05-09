"""Document category classification (Gemini).

Returns ``Category.NOVEL`` or ``Category.EDUCATIONAL`` from a text excerpt.
Falls back to ``Category.UNKNOWN`` if Gemini is unavailable, leaving it to the
user to set the category manually via PATCH /api/jobs/{id}/category.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.db.models import Category
from app.services.gemini import GeminiUnavailable, generate_json

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "classify.txt"


@dataclass(frozen=True)
class ClassificationResult:
    category: Category
    confidence: float
    reason: str
    auto_detected: bool  # True when Gemini decided; False when we fell back to UNKNOWN


_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["NOVEL", "EDUCATIONAL"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["category"],
}


def classify_document(excerpt: str) -> ClassificationResult:
    """Classify a document from its leading text excerpt."""
    if not excerpt.strip():
        return ClassificationResult(Category.UNKNOWN, 0.0, "empty excerpt", auto_detected=False)

    s = get_settings()
    system = _PROMPT_PATH.read_text(encoding="utf-8")
    try:
        resp = generate_json(
            model=s.gemini_utility_model,
            system_instruction=system,
            user_text=excerpt,
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.0,
        )
    except GeminiUnavailable as e:
        log.warning("classification skipped, Gemini unavailable: %s", e)
        return ClassificationResult(
            Category.UNKNOWN, 0.0, f"gemini unavailable: {e}", auto_detected=False
        )

    payload = resp.payload
    raw_category = (payload.get("category") or "").upper()
    try:
        category = Category(raw_category)
    except ValueError:
        log.warning("classifier returned unexpected category: %r", raw_category)
        return ClassificationResult(
            Category.UNKNOWN, 0.0, f"unexpected category: {raw_category!r}", auto_detected=False
        )

    return ClassificationResult(
        category=category,
        confidence=float(payload.get("confidence") or 0.0),
        reason=str(payload.get("reason") or ""),
        auto_detected=True,
    )
