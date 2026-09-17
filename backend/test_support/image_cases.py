"""Synthetic camera / reject images for intake and API tests."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATARACT = (
    PROJECT_ROOT
    / "dataset"
    / "Eye Disease detection"
    / "cataract"
    / "_114_5711178.jpg"
)


def _jpeg_bytes(image: Image.Image, *, quality: int = 92, exif: Image.Exif | None = None) -> bytes:
    buffer = io.BytesIO()
    kwargs = {"format": "JPEG", "quality": quality}
    if exif is not None:
        kwargs["exif"] = exif
    image.convert("RGB").save(buffer, **kwargs)
    return buffer.getvalue()


def fundus_camera_style_jpeg(source: Path | None = None) -> bytes:
    """Phone-style capture: fundus print on a larger background, EXIF orientation."""
    fundus = Image.open(source or CATARACT).convert("RGB")
    canvas = Image.new("RGB", (fundus.width * 2, fundus.height * 2), (28, 32, 36))
    offset = ((canvas.width - fundus.width) // 2, (canvas.height - fundus.height) // 2)
    canvas.paste(fundus, offset)
    rotated = canvas.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 8
    return _jpeg_bytes(rotated, exif=exif)


def blurry_fundus_jpeg(source: Path | None = None) -> bytes:
    image = Image.open(source or CATARACT).convert("RGB")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=18))
    return _jpeg_bytes(blurred, quality=70)


def unsupported_scene_jpeg() -> bytes:
    """Outdoor-like scene: not a retinal photograph."""
    image = Image.new("RGB", (480, 360), (92, 168, 228))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 220, 480, 360), fill=(46, 128, 52))
    draw.ellipse((320, 36, 400, 116), fill=(252, 220, 80))
    return _jpeg_bytes(image)


def external_eye_jpeg() -> bytes:
    """Synthetic external-eye / face-like photo: sclera + iris, not fundus."""
    image = Image.new("RGB", (480, 360), (214, 176, 156))
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 110, 410, 250), fill=(248, 246, 242))
    draw.ellipse((185, 132, 295, 228), fill=(78, 112, 86))
    draw.ellipse((215, 158, 265, 208), fill=(18, 18, 18))
    return _jpeg_bytes(image)


def fundus_webp(source: Path | None = None) -> bytes:
    image = Image.open(source or CATARACT).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=95)
    return buffer.getvalue()
