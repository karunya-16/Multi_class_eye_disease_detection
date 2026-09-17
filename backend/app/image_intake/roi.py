"""Safe cropping / ROI extraction for camera photographs of fundus images."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.image_intake.fundus_cues import fundus_score


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    if (x1 - x0) < 32 or (y1 - y0) < 32:
        return None
    return x0, y0, x1, y1


def _crop_with_padding(image: Image.Image, box: tuple[int, int, int, int], pad_ratio: float = 0.04) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = box
    pad_x = int((x1 - x0) * pad_ratio)
    pad_y = int((y1 - y0) * pad_ratio)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    return image.crop((x0, y0, x1, y1))


def crop_dark_borders(image: Image.Image) -> Image.Image | None:
    """Remove near-black padding typical of fundus cameras and some phone photos."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    box = _bbox_from_mask(gray > 12.0)
    if box is None:
        return None
    cropped = _crop_with_padding(image, box)
    if cropped.size[0] < 64 or cropped.size[1] < 64:
        return None
    return cropped


def crop_colored_retina_region(image: Image.Image) -> Image.Image | None:
    """Crop the reddish/orange blob when a fundus photo sits on a larger background."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = arr.mean(axis=2)
    mask = (red > 40.0) & (red + 4.0 >= green) & (gray > 16.0) & (gray < 245.0)
    box = _bbox_from_mask(mask)
    if box is None:
        return None
    cropped = _crop_with_padding(image, box, pad_ratio=0.06)
    if cropped.size[0] < 64 or cropped.size[1] < 64:
        return None
    return cropped


def candidate_images(image: Image.Image) -> list[Image.Image]:
    """Original plus optional ROI crops; the workflow picks the best fundus-like crop."""
    candidates = [image]
    for cropper in (crop_dark_borders, crop_colored_retina_region):
        cropped = cropper(image)
        if cropped is None:
            continue
        if cropped.size == image.size:
            continue
        candidates.append(cropped)
    return candidates


def select_best_roi(image: Image.Image) -> Image.Image:
    best = image
    best_score = fundus_score(image)
    for candidate in candidate_images(image)[1:]:
        score = fundus_score(candidate)
        if score > best_score + 0.02:
            best = candidate
            best_score = score
    return best
