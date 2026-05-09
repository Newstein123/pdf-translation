"""Token counting using tiktoken.

We don't have a Gemini-specific tokenizer locally, so we use ``cl100k_base``
(OpenAI's GPT-4 tokenizer) as a stable approximation. The exact count doesn't
matter for chunk sizing — it just needs to be consistent across calls.
"""

from __future__ import annotations

import logging
from functools import lru_cache

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        log.warning("tiktoken unavailable (%s); falling back to char-based estimate", e)
        return None


def count_tokens(text: str) -> int:
    """Return an approximate token count for ``text``."""
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        # ~4 chars/token is a reasonable rule-of-thumb fallback.
        return max(1, len(text) // 4)
    return len(enc.encode(text))
