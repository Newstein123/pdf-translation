"""Cover image upload + normalization (M16).

The user uploads a JPG / PNG / WEBP via the API; this module:

  * Validates the file is actually an image (Pillow open succeeds)
  * Re-encodes to JPEG with EXIF orientation applied
  * Resizes to a reasonable max dimension so the resulting PDF stays small
  * Saves to ``storage/intermediate/<job_id>/cover.jpg``
  * Returns the path so the caller can persist it on the Job row

The renderer side (``pdf_template`` / ``rebuild_pdf``) reads the saved
file and prepends a single full-page cover image to the exported PDF.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, ImageOps

from app.config import get_settings

log = logging.getLogger(__name__)

# Max dimension on the long edge after resize. Cover pages don't need to be
# huge — at 1600px on the long edge, even a full A5 page renders crisply
# at 200+ DPI without ballooning the output PDF.
_MAX_LONG_EDGE_PX = 1600
# JPEG quality. 88 is the sweet spot between visible artifacts and file size
# for photographic / illustrated covers.
_JPEG_QUALITY = 88
# Allowed input formats. Pillow can read more, but limiting up-front means
# the user gets a clear error rather than a surprising decode at runtime.
_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"}


class CoverImageError(RuntimeError):
    pass


def save_cover_image(*, job_id: str, raw_bytes: bytes) -> Path:
    """Validate, normalize, and save the user-supplied cover image.

    Returns the absolute path to the saved JPEG. Raises
    ``CoverImageError`` with a user-readable message on:
      * Empty / malformed input
      * Unsupported format
      * Decode failure
    """
    if not raw_bytes:
        raise CoverImageError("empty file")

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except Exception as e:
        raise CoverImageError(f"could not decode image: {e}") from e

    fmt = (img.format or "").upper()
    if fmt not in _ALLOWED_FORMATS:
        raise CoverImageError(
            f"unsupported image format {fmt!r}; expected one of "
            f"{sorted(_ALLOWED_FORMATS)}"
        )

    # Apply EXIF orientation (phone photos arrive sideways without this).
    img = ImageOps.exif_transpose(img)
    # Convert to RGB — JPEG can't store alpha; transparent pixels become
    # white via paste-on-white-background.
    if img.mode in {"RGBA", "LA", "P"}:
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize so the long edge is at most _MAX_LONG_EDGE_PX. Don't upscale.
    long_edge = max(img.width, img.height)
    if long_edge > _MAX_LONG_EDGE_PX:
        scale = _MAX_LONG_EDGE_PX / long_edge
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Sanity check: refuse uploads that flatten to (near-)uniform white.
    # Most common cause: a fully-transparent PNG. Without this guard the
    # user gets a confusing white cover page on every export. We sample
    # ~50 pixels across the image and reject if 99%+ are near-white —
    # that's a strong signal the foreground content didn't make it
    # through the flatten step.
    if _looks_uniform_white(img):
        raise CoverImageError(
            "uploaded image appears blank / fully transparent after "
            "flattening. PNGs with a transparent background can produce "
            "this when the alpha channel is empty. Try saving as JPG or "
            "exporting the PNG with the content baked onto a visible "
            "background."
        )

    settings = get_settings()
    out_dir = settings.intermediate_dir / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cover.jpg"
    img.save(out_path, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    log.info(
        "saved cover image for job %s: %s (%dx%d, %d bytes)",
        job_id,
        out_path,
        img.width,
        img.height,
        out_path.stat().st_size,
    )
    return out_path


def _looks_uniform_white(img) -> bool:
    """Return True if the image is overwhelmingly near-white pixels.

    Samples a 7x7 grid of pixels (49 total) — enough to catch genuinely
    blank uploads while being O(1) in image size. Threshold: a pixel
    counts as "near-white" if every channel is ≥ 245. Image is "uniform
    white" if ≥ 98% of sampled pixels meet that bar."""
    w, h = img.size
    if w < 4 or h < 4:
        return True  # too small to be a real cover anyway
    samples = []
    for y_idx in range(7):
        for x_idx in range(7):
            x = (w - 1) * x_idx // 6
            y = (h - 1) * y_idx // 6
            px = img.getpixel((x, y))
            if isinstance(px, int):
                px = (px, px, px)
            samples.append(px[:3])
    near_white = sum(1 for r, g, b in samples if r >= 245 and g >= 245 and b >= 245)
    return near_white >= int(len(samples) * 0.98)


def delete_cover_image(*, cover_path: str | Path | None) -> None:
    """Best-effort cleanup. Missing files are not an error — the column
    might already point at a path that's been swept by retention."""
    if not cover_path:
        return
    p = Path(cover_path)
    try:
        p.unlink(missing_ok=True)
    except OSError:
        log.exception("failed to delete cover image at %s", p)
