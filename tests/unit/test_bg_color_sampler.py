"""Unit tests for the bbox background-color sampler.

The sampler's job: given a rendered page and a text bbox, return the dominant
background color (so the rebuilder can fill that bbox correctly instead of
always stamping white). Anti-aliased glyph edges should NOT fool the
histogram.
"""
from __future__ import annotations

import numpy as np

from app.services.extraction import _BG_CONFIDENCE_THRESHOLD, _dominant_bg_color


def _make_pix_array(
    width: int, height: int, bg: tuple[int, int, int]
) -> np.ndarray:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = bg
    return arr


def _put_text_strokes(
    arr: np.ndarray,
    bbox_px: tuple[int, int, int, int],
    color: tuple[int, int, int],
    coverage: float,
) -> None:
    """Scatter ``color`` pixels into the bbox to simulate text glyph strokes.
    Order-independent because the histogram is."""
    x0, y0, x1, y1 = bbox_px
    h, w = y1 - y0, x1 - x0
    rng = np.random.default_rng(seed=42)
    mask = rng.random((h, w)) < coverage
    arr[y0:y1, x0:x1, :][mask] = color


def test_white_page_returns_pure_white() -> None:
    """Pure-white pages must produce #ffffff, not #f8f8f8 (the previous
    bin-center result). Even a 7-unit difference shows up as a faint gray
    rectangle around every text block on a #ffffff page."""
    arr = _make_pix_array(400, 200, (255, 255, 255))
    _put_text_strokes(arr, (50, 50, 350, 150), (0, 0, 0), coverage=0.30)
    color = _dominant_bg_color(
        bbox=(50, 50, 350, 150),
        pix_array=arr,
        page_width=400,
        page_height=200,
    )
    # Median of bin-15 pixels = 255 since every pixel in that bin is exactly 255.
    assert color == "#ffffff"


def test_blue_panel_returns_exact_blue() -> None:
    arr = _make_pix_array(400, 200, (255, 255, 255))
    arr[40:160, 40:360, :] = (50, 100, 200)
    _put_text_strokes(arr, (50, 50, 350, 150), (255, 255, 255), coverage=0.30)
    color = _dominant_bg_color(
        bbox=(50, 50, 350, 150),
        pix_array=arr,
        page_width=400,
        page_height=200,
    )
    # Median of bin pixels = (50, 100, 200) — exact original color.
    assert color == "#3264c8"


def test_pale_tint_returns_exact_tint() -> None:
    """A very-pale-blue panel (#e9f3f9 — common table cell shading) now
    returns its exact original color, not a quantized approximation."""
    arr = _make_pix_array(400, 200, (233, 243, 249))
    _put_text_strokes(arr, (50, 50, 350, 150), (50, 50, 50), coverage=0.30)
    color = _dominant_bg_color(
        bbox=(50, 50, 350, 150),
        pix_array=arr,
        page_width=400,
        page_height=200,
    )
    assert color == "#e9f3f9"


def test_returns_none_when_below_confidence() -> None:
    """A bbox over an image-y region (no clearly dominant color) should yield
    None so the rebuild falls back to white instead of guessing."""
    rng = np.random.default_rng(seed=1)
    arr = rng.integers(0, 256, size=(200, 400, 3), dtype=np.uint8)
    color = _dominant_bg_color(
        bbox=(50, 50, 350, 150),
        pix_array=arr,
        page_width=400,
        page_height=200,
    )
    assert color is None


def test_returns_none_when_pixmap_missing() -> None:
    color = _dominant_bg_color(
        bbox=(0, 0, 100, 100),
        pix_array=None,
        page_width=400,
        page_height=200,
    )
    assert color is None


def test_returns_none_for_invalid_bbox() -> None:
    arr = _make_pix_array(100, 100, (255, 255, 255))
    color = _dominant_bg_color(
        bbox=(50, 50, 50, 50),  # zero-area
        pix_array=arr,
        page_width=100,
        page_height=100,
    )
    assert color is None


def test_handles_dpi_scaled_pixmap() -> None:
    """When the pixmap is rendered at higher resolution than PDF points,
    bbox coords scale accordingly."""
    arr = _make_pix_array(400, 200, (255, 255, 255))
    arr[:, :200, :] = (200, 50, 50)
    color = _dominant_bg_color(
        bbox=(20, 20, 80, 80),  # PDF points → maps to pixel (40,40,160,160) at 2x
        pix_array=arr,
        page_width=200,
        page_height=100,
    )
    # Median of bin pixels = (200, 50, 50)
    assert color == "#c83232"


def test_dense_paragraph_still_picks_background() -> None:
    """Even when text covers ~50% of the bbox, quantization-absorbed AA
    edges keep the bg color as the histogram dominant; median over bin
    pixels then returns the exact original color."""
    arr = _make_pix_array(400, 200, (240, 235, 220))  # cream paper
    _put_text_strokes(arr, (20, 20, 380, 180), (30, 30, 30), coverage=0.50)
    color = _dominant_bg_color(
        bbox=(20, 20, 380, 180),
        pix_array=arr,
        page_width=400,
        page_height=200,
    )
    # Cream — exact original (240, 235, 220).
    assert color == "#f0ebdc"
