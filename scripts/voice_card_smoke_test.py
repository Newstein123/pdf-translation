#!/usr/bin/env python3
"""Smoke test: translate the same English passage three ways and compare.

Calls Gemini directly via the existing ``translation.translate_chunk`` to
verify that voice-card injection actually shapes the output. Three variants:

  1. ShweOoDaung's voice card → expect formal classical Burmese register
  2. Mya Than Tint's voice card → expect conversational modern Burmese
  3. No voice card (control) → default model behavior

Prints the three Burmese translations side-by-side, plus token usage and a
USD estimate so you can see the cost per call.

Cost: typically <$0.02 total (three short Gemini Pro calls).

Usage::

    python scripts/voice_card_smoke_test.py

Requires ``GEMINI_API_KEY`` in the environment / .env file.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.db.models import Category  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services import translation, voice_card  # noqa: E402

log = logging.getLogger("voice_card_smoke")

# Public-domain Austen prose. Third-person past, gentle narrative — neutral
# enough that both voices can plausibly handle it, distinct enough that a
# voice-anchored translation should differ visibly from the control.
TEST_PASSAGE = (
    "Mr. Bennet was among the earliest of those who waited on Mr. Bingley. "
    "He had always intended to visit him, though to the last always assuring "
    "his wife that he should not go; and till the evening after the visit "
    "was paid, she had no knowledge of it."
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if not settings.gemini_api_key:
        log.error("GEMINI_API_KEY is not set in the environment / .env")
        return 2

    blocks = [{"block_id": "b0", "source": TEST_PASSAGE}]

    variants = [
        ("ShweOoDaung (formal classical)", "ShweOoDaung"),
        ("Mya Than Tint (modern conversational)", "မြသန်းတင့်"),
        ("No voice card (control)", None),
    ]

    print("=" * 70)
    print("VOICE CARD SMOKE TEST")
    print("=" * 70)
    print(f"Source ({len(TEST_PASSAGE.split())} words):")
    print(TEST_PASSAGE)
    print()
    print(f"Model: {settings.gemini_translation_model}")
    print()

    total_input = 0
    total_output = 0
    results: list[dict[str, Any]] = []

    with SessionLocal() as session:
        for label, translator in variants:
            print("-" * 70)
            print(f"VARIANT: {label}")
            print("-" * 70)

            voice_text: str | None = None
            if translator is not None:
                card = voice_card.get_active_card(session, translator)
                if card is None:
                    print(f"  WARNING: no active voice card for {translator!r}; skipping.")
                    print()
                    continue
                voice_text = voice_card.compose_voice_card_block(card)

            system_prompt = translation.compose_translate_system_prompt(
                Category.NOVEL, profile=None, voice_card_text=voice_text
            )

            try:
                result = translation.translate_chunk(
                    blocks=blocks,
                    glossary=[],
                    name_registry=[],
                    previous_summary=None,
                    category=Category.NOVEL,
                    style_profile=None,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                print()
                continue

            if not result.blocks:
                print("  WARNING: empty translation")
                print()
                continue

            burmese = result.blocks[0].translated
            print(burmese)
            print()
            print(
                f"  tokens: prompt={result.prompt_tokens}, "
                f"output={result.output_tokens}, total={result.total_tokens}"
            )
            total_input += result.prompt_tokens or 0
            total_output += result.output_tokens or 0
            results.append({"label": label, "burmese": burmese, "result": result})
            print()

    # Cost estimate using the configured Pro rates.
    pro_in_per_m = settings.gemini_pro_input_usd_per_m
    pro_out_per_m = settings.gemini_pro_output_usd_per_m
    est_usd = (
        total_input * pro_in_per_m / 1_000_000
        + total_output * pro_out_per_m / 1_000_000
    )

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Variants run: {len(results)}")
    print(f"Total input tokens: {total_input}")
    print(f"Total output tokens: {total_output}")
    print(f"Estimated USD cost: ${est_usd:.4f}")
    print()
    print("Compare the three outputs above — look for:")
    print("  * Sentence length (formal voice = longer sentences)")
    print("  * Verb endings (formal voice = ေလ၏ / ေပ၏ / သည်; modern = တယ် / ပြီ)")
    print("  * Honorifics (formal = ကျနော်/ခင်ဗျား; modern = ကျွန်တော် with familial)")
    print("  * Particles (modern voice = ပေါ့/လေ/ကွာ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
