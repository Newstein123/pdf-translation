#!/usr/bin/env bash
# Fetch Burmese TTFs into fonts/. Idempotent.
#
# Bundles two families:
#   * Noto Sans Myanmar (sans-serif, modern feel) — fallback / web reading
#   * Padauk (SIL, Myanmar Unicode 5.1+, traditional book-style feel) —
#     preferred for the printed-book template. ``get_burmese_fonts()`` picks
#     Pyidaungsu → Noto Sans Myanmar → Padauk in that order; the printed-book
#     template overrides via ``settings.template_body_font``.
#
# We use static (non-variable) TTFs because PyMuPDF cannot embed variable
# fonts cleanly — glyphs render but text is unextractable, which breaks
# both our test suite and downstream copy/paste.
set -euo pipefail

cd "$(dirname "$0")/.."
FONT_DIR="fonts"
mkdir -p "$FONT_DIR"

NOTO_BASE="https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansMyanmar/unhinted/ttf"
declare -a TARGETS_NOTO=(
  "NotoSansMyanmar-Regular.ttf"
  "NotoSansMyanmar-Bold.ttf"
)

# Padauk: SIL's reference Myanmar font. We pull from Google Fonts' mirror
# rather than SIL's official release tarballs (versioned, not stable URLs)
# or the silnrsi GitHub repo (path moves between releases). Google Fonts'
# OFL tree is the most stable public mirror for static Padauk TTFs.
PADAUK_BASE="https://github.com/google/fonts/raw/main/ofl/padauk"
declare -a TARGETS_PADAUK=(
  "Padauk-Regular.ttf"
  "Padauk-Bold.ttf"
)

download_one() {
  local base="$1"
  local name="$2"
  local dest="$FONT_DIR/$name"
  if [[ -f "$dest" ]]; then
    if [[ $(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest") -gt 50000 ]]; then
      echo "Already present: $dest"
      return 0
    fi
  fi
  echo "Downloading $dest"
  if ! curl -fsSL -o "$dest" "$base/$name"; then
    echo "  !! download failed for $name (continuing without it)"
    rm -f "$dest"
    return 1
  fi
}

for name in "${TARGETS_NOTO[@]}"; do
  download_one "$NOTO_BASE" "$name" || true
done
for name in "${TARGETS_PADAUK[@]}"; do
  download_one "$PADAUK_BASE" "$name" || true
done

echo "Done. Fonts available in $FONT_DIR/"
ls -lh "$FONT_DIR/"
