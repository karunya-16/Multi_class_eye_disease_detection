"""Decode uploaded bytes: EXIF orientation correction and RGB conversion."""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from app.image_intake.constants import INVALID_IMAGE_MESSAGE, UNSUPPORTED_FILE_MESSAGE

_HEIF_REGISTERED = False


def _register_heif_opener() -> None:
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception:
        pass
    _HEIF_REGISTERED = True


def decode_rgb(data: bytes) -> Image.Image:
    """Open an image, apply EXIF orientation, and convert to RGB.

    Does not resize and does not divide by 255. Those steps stay in
    ``app.model_service.preprocess``.
    """
    _register_heif_opener()
    try:
        with Image.open(io.BytesIO(data)) as src:
            image = ImageOps.exif_transpose(src)
            if image is None:
                image = src
            rgb = image.convert("RGB")
            rgb.load()
            return rgb
    except UnidentifiedImageError as exc:
        raise ValueError(INVALID_IMAGE_MESSAGE) from exc
    except OSError as exc:
        message = str(exc).lower()
        if "heic" in message or "heif" in message:
            raise ValueError(UNSUPPORTED_FILE_MESSAGE) from exc
        raise ValueError(INVALID_IMAGE_MESSAGE) from exc


def orientation_was_corrected(data: bytes, decoded: Image.Image) -> bool:
    """True when EXIF transpose changed the pixel layout."""
    try:
        with Image.open(io.BytesIO(data)) as src:
            raw = src.convert("RGB")
            if raw.size != decoded.size:
                return True
            return False
    except Exception:
        return True
