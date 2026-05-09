"""Font resolution for Burmese-rendering PDFs and EPUBs.

We prefer a bundled Noto Sans Myanmar (Regular + Bold) under ``fonts/``. If it
isn't present, we log a warning and fall back to PyMuPDF's built-in font, which
can render Latin text but won't produce correct Burmese glyphs. The fallback is
intentional: the pipeline still completes end-to-end so the user can verify
plumbing, and ``scripts/download_fonts.sh`` provides the proper fonts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Repo-root/fonts/
_FONTS_DIR = Path(__file__).resolve().parents[2] / "fonts"

# Filenames we look for — set by scripts/download_fonts.sh.
# Order is the fallback chain: prefer Pyidaungsu (Myanmar government standard
# with strong native shaping), then Noto Sans Myanmar, then Padauk.
_REGULAR_CANDIDATES = (
    "Pyidaungsu-2.5_Regular.ttf",
    "Pyidaungsu-Regular.ttf",
    "NotoSansMyanmar-Regular.ttf",
    "Padauk-Regular.ttf",
)
_BOLD_CANDIDATES = (
    "Pyidaungsu-2.5_Bold.ttf",
    "Pyidaungsu-Bold.ttf",
    "NotoSansMyanmar-Bold.ttf",
    "Padauk-Bold.ttf",
)


@dataclass(frozen=True)
class BurmeseFontSet:
    regular_path: Path | None
    bold_path: Path | None
    family_name: str  # the family advertised in EPUB CSS

    @property
    def has_burmese(self) -> bool:
        return self.regular_path is not None


def _first_existing(candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        p = _FONTS_DIR / name
        if p.exists():
            return p
    return None


def get_burmese_fonts(preferred_name: str | None = None) -> BurmeseFontSet:
    """Resolve a Burmese font set, optionally biased toward a named family.

    ``preferred_name`` accepts ``"Pyidaungsu"``, ``"Padauk"``, or
    ``"Noto Sans Myanmar"`` (case-insensitive). When provided and that
    family is installed, it wins regardless of the default ordering. This
    is what the printed-book template uses to pick a more book-style font
    (Padauk) while leaving the layout-preserving rebuild on its default
    fallback chain.
    """
    if preferred_name:
        forced = _resolve_preferred_family(preferred_name)
        if forced is not None:
            return forced

    regular = _first_existing(_REGULAR_CANDIDATES)
    bold = _first_existing(_BOLD_CANDIDATES)
    if regular is None:
        log.warning(
            "No Burmese font found in %s. Run scripts/download_fonts.sh to install "
            "Noto Sans Myanmar. Burmese glyphs will not render correctly without it.",
            _FONTS_DIR,
        )
        return BurmeseFontSet(None, None, "sans-serif")
    if "Pyidaungsu" in regular.name:
        family = "Pyidaungsu"
    elif "Noto" in regular.name:
        family = "Noto Sans Myanmar"
    else:
        family = "Padauk"
    return BurmeseFontSet(regular_path=regular, bold_path=bold, family_name=family)


def _resolve_preferred_family(name: str) -> BurmeseFontSet | None:
    """Look up a specific font family by name. Returns ``None`` if not
    installed so the caller can fall back to the default chain."""
    n = name.strip().lower()
    if "pyidaungsu" in n:
        candidates_r = ("Pyidaungsu-2.5_Regular.ttf", "Pyidaungsu-Regular.ttf")
        candidates_b = ("Pyidaungsu-2.5_Bold.ttf", "Pyidaungsu-Bold.ttf")
        family = "Pyidaungsu"
    elif "padauk" in n:
        candidates_r = ("Padauk-Regular.ttf",)
        candidates_b = ("Padauk-Bold.ttf",)
        family = "Padauk"
    elif "noto" in n:
        candidates_r = ("NotoSansMyanmar-Regular.ttf",)
        candidates_b = ("NotoSansMyanmar-Bold.ttf",)
        family = "Noto Sans Myanmar"
    else:
        return None
    regular = _first_existing(candidates_r)
    if regular is None:
        return None
    bold = _first_existing(candidates_b)
    return BurmeseFontSet(regular_path=regular, bold_path=bold, family_name=family)


def fonts_dir() -> Path:
    return _FONTS_DIR
