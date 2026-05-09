"""Thin wrapper around the ``google-genai`` SDK.

Centralizes:
  - lazy client construction with the API key from settings
  - graceful degradation when no key is set (raises ``GeminiUnavailable``)
  - structured-JSON helper used by classifier + (later) translator
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)

_client: Any | None = None

# Retry tuning for transient errors (rate limits, 5xx, transient network).
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 30.0

# Substrings that mark errors as worth retrying. ``GeminiUnavailable`` raised
# *into* this layer (e.g. invalid JSON) is NOT retried.
_TRANSIENT_HINTS = (
    "rate limit",
    "resource_exhausted",
    "deadline_exceeded",
    "unavailable",
    "internal",
    "timeout",
    "503",
    "504",
    "429",
)


@dataclass(frozen=True)
class GeminiJsonResponse:
    payload: dict[str, Any]
    total_tokens: int
    prompt_tokens: int = 0
    output_tokens: int = 0
    # Subset of ``prompt_tokens`` Gemini billed at the cached rate. Populated
    # when the call referenced a context-cache handle; 0 otherwise. Surfaced
    # so the cost estimator can split the input-tokens bucket between cached
    # (~25% of standard input rate) and uncached.
    cached_input_tokens: int = 0


class GeminiUnavailable(RuntimeError):
    """Raised when Gemini is not configured (no API key) or the SDK fails."""


class GeminiModelMissing(GeminiUnavailable):
    """The configured Gemini model returned 404 / NOT_FOUND.

    Distinguished from generic ``GeminiUnavailable`` so callers can fast-fail
    the entire job — a missing/deprecated model affects ALL chunks identically,
    so retrying chunk-by-chunk just burns budget."""


class GeminiAccessDenied(GeminiUnavailable):
    """The Gemini call returned 403 / PERMISSION_DENIED.

    Same fast-fail rationale as ``GeminiModelMissing``: a project-level block,
    expired/revoked key, or quota exhaustion affects every subsequent call —
    retrying just delays the inevitable. Surfaces a clear remediation
    message pointing the operator at the API key dashboard."""


def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client
    s = get_settings()
    if not s.gemini_api_key:
        raise GeminiUnavailable("GEMINI_API_KEY is not set")
    try:
        from google import genai

        _client = genai.Client(api_key=s.gemini_api_key)
        return _client
    except Exception as e:
        raise GeminiUnavailable(f"failed to create Gemini client: {e}") from e


def reset_client() -> None:
    """Drop the cached client. Useful when settings change (tests)."""
    global _client
    _client = None


def generate_json(
    *,
    model: str,
    system_instruction: str,
    user_text: str,
    response_schema: dict[str, Any] | None = None,
    temperature: float = 0.1,
) -> GeminiJsonResponse:
    """Call Gemini with a system instruction + user text, expecting JSON back.

    Retries with exponential backoff + jitter on transient failures (rate
    limits, 5xx, deadline exceeded). Non-transient failures (auth, malformed
    response) raise ``GeminiUnavailable`` immediately.
    """
    client = _get_client()
    from google.genai import types

    config_kwargs: dict[str, Any] = {
        "system_instruction": system_instruction,
        "temperature": temperature,
        "response_mime_type": "application/json",
    }
    if response_schema is not None:
        config_kwargs["response_schema"] = response_schema

    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_text,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = (response.text or "").strip()
            if not text:
                raise GeminiUnavailable("empty response from Gemini")
            payload = json.loads(text)
            prompt_t, output_t, total_t, cached_t = _extract_token_counts(response)
            return GeminiJsonResponse(
                payload=payload,
                total_tokens=total_t,
                prompt_tokens=prompt_t,
                output_tokens=output_t,
                cached_input_tokens=cached_t,
            )
        except json.JSONDecodeError as e:
            log.warning("Gemini returned non-JSON: %s", e)
            raise GeminiUnavailable(f"non-JSON response: {e}") from e
        except Exception as e:  # noqa: BLE001 — we sort transient vs not below
            last_err = e
            if _is_model_missing(e):
                log.error(
                    "Gemini model %r is not available — failing fast (no retries)", model
                )
                raise GeminiModelMissing(
                    f"model {model!r} not found: {e}. Update GEMINI_TRANSLATION_MODEL/"
                    f"GEMINI_UTILITY_MODEL in .env to a current model name."
                ) from e
            if _is_access_denied(e):
                log.error(
                    "Gemini returned 403 PERMISSION_DENIED — failing fast (no retries). "
                    "Key may be revoked, project suspended, or quota exhausted."
                )
                raise GeminiAccessDenied(
                    f"403 PERMISSION_DENIED: {e}. Verify GEMINI_API_KEY is active "
                    "at https://aistudio.google.com/apikey, check project billing/"
                    "quotas, then restart the worker."
                ) from e
            if attempt >= _MAX_RETRIES or not _is_transient(e):
                log.exception("Gemini call failed (final attempt %d)", attempt)
                raise GeminiUnavailable(str(e)) from e
            backoff = min(
                _BASE_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 0.5),
                _MAX_BACKOFF_SECONDS,
            )
            log.warning(
                "Gemini call transient failure (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                _MAX_RETRIES + 1,
                e,
                backoff,
            )
            time.sleep(backoff)

    # Defensive: loop should always return or raise above.
    raise GeminiUnavailable(str(last_err) if last_err else "exhausted retries")


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _TRANSIENT_HINTS)


def _is_model_missing(exc: BaseException) -> bool:
    """404 from the SDK, regardless of message phrasing."""
    msg = str(exc).lower()
    if "404" in msg and ("not_found" in msg or "not found" in msg):
        return True
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return code == 404


def _is_access_denied(exc: BaseException) -> bool:
    """403 PERMISSION_DENIED from the SDK — auth/billing/quota problem.

    Covers explicit "permission_denied" status strings and "access blocked"
    phrases the SDK occasionally returns instead of a clean 403. Quota
    exhaustion is intentionally NOT included here — that surfaces as 429
    (rate-limited) and IS retriable."""
    msg = str(exc).lower()
    if "403" in msg and ("permission_denied" in msg or "denied" in msg):
        return True
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return code == 403


def _extract_total_tokens(response: Any) -> int:
    """Best-effort extraction of total token count from a generate_content response."""
    _, _, total, _ = _extract_token_counts(response)
    return total


def _extract_token_counts(response: Any) -> tuple[int, int, int, int]:
    """Return ``(prompt_tokens, output_tokens, total_tokens, cached_input_tokens)``.

    Falls back to ``prompt + candidates`` when the SDK only reports the parts.
    ``cached_input_tokens`` is the subset of ``prompt_tokens`` Gemini billed
    at the cached rate (only non-zero when the call referenced a context
    cache); attribute name is ``cached_content_token_count`` in the
    google-genai SDK. We accept ``int | None`` and clamp to ``[0, prompt]``
    so a misbehaving SDK can't push the uncached bucket negative.
    """
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return 0, 0, 0, 0
    prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
    output = int(getattr(meta, "candidates_token_count", 0) or 0)
    total = getattr(meta, "total_token_count", None)
    if total is None:
        total = prompt + output
    cached = int(getattr(meta, "cached_content_token_count", 0) or 0)
    cached = max(0, min(cached, prompt))
    return prompt, output, int(total), cached
