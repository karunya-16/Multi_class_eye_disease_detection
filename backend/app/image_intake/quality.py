"""Brightness, blur, and size checks for uploaded eye photographs."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from app.image_intake.constants import (
    CLEAR_IMAGE_MESSAGE,
    MAX_MEAN_BRIGHTNESS,
    MIN_HIGHFREQ,
    MIN_MEAN_BRIGHTNESS,
    MIN_SIDE_PX,
    MIN_STRUCTURE_STD,
)
from app.image_intake.exceptions import UnsuitableImageError


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def high_frequency_energy(image: Image.Image) -> float:
    """Mean absolute residual after a small Gaussian blur (fine detail)."""
    gray = _gray(image)
    blurred = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=2)).convert("L"),
        dtype=np.float32,
    )
    return float(np.mean(np.abs(gray - blurred)))


def brightness_stats(image: Image.Image) -> tuple[float, float]:
    gray = _gray(image)
    return float(gray.mean()), float(gray.std())


def check_quality(image: Image.Image) -> None:
    width, height = image.size
    if min(width, height) < MIN_SIDE_PX:
        raise UnsuitableImageError(CLEAR_IMAGE_MESSAGE)
    mean, std = brightness_stats(image)
    if mean < MIN_MEAN_BRIGHTNESS or mean > MAX_MEAN_BRIGHTNESS:
        raise UnsuitableImageError(CLEAR_IMAGE_MESSAGE)
    if std < MIN_STRUCTURE_STD:
        raise UnsuitableImageError(CLEAR_IMAGE_MESSAGE)
    if high_frequency_energy(image) < MIN_HIGHFREQ:
        raise UnsuitableImageError(CLEAR_IMAGE_MESSAGE)
