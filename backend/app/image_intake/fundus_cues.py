"""Heuristic cues that a photograph looks like a retinal fundus image.

The trained model was validated on fundus photographs, not on selfies or
arbitrary smartphone pictures of the external eye. These scores decide
whether an upload may be forwarded to that model.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.image_intake.constants import CLEAR_IMAGE_MESSAGE, MIN_FUNDUS_SCORE
from app.image_intake.exceptions import UnsuitableImageError


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _corner_and_center(gray: np.ndarray) -> tuple[float, float]:
    height, width = gray.shape
    margin = max(2, min(height, width) // 8)
    corners = np.concatenate(
        [
            gray[:margin, :margin].ravel(),
            gray[:margin, -margin:].ravel(),
            gray[-margin:, :margin].ravel(),
            gray[-margin:, -margin:].ravel(),
        ]
    )
    cy1, cy2 = height // 2 - max(height // 10, 1), height // 2 + max(height // 10, 1)
    cx1, cx2 = width // 2 - max(width // 10, 1), width // 2 + max(width // 10, 1)
    center = gray[cy1:cy2, cx1:cx2]
    return float(corners.mean()), float(center.mean())


def fundus_features(image: Image.Image) -> dict[str, float]:
    arr = _rgb_array(image)
    gray = arr.mean(axis=2)
    corner_mean, center_mean = _corner_and_center(gray)
    red, green, blue = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    vignette = center_mean - corner_mean
    white_frac = float(np.mean((arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200)))
    dark_corner = float(corner_mean < 25.0)
    return {
        "vignette": float(vignette),
        "red_minus_green": float(red - green),
        "red_minus_blue": float(red - blue),
        "corner_mean": float(corner_mean),
        "center_mean": float(center_mean),
        "std": float(gray.std()),
        "white_frac": white_frac,
        "dark_corner": dark_corner,
    }


def fundus_score(image: Image.Image) -> float:
    feats = fundus_features(image)
    vignette = min(max(feats["vignette"] / 80.0, 0.0), 1.0)
    red_dom = min(max(feats["red_minus_green"] / 35.0, 0.0), 1.0)
    circular = 1.0 if feats["corner_mean"] < 28.0 and feats["center_mean"] > 35.0 else 0.0
    if feats["vignette"] >= 18.0:
        circular = max(circular, 0.7)
    structure = min(max(feats["std"] / 40.0, 0.0), 1.0)
    white_penalty = min(feats["white_frac"] * 1.5, 1.0)
    score = 0.42 * vignette + 0.22 * red_dom + 0.26 * circular + 0.10 * structure - 0.35 * white_penalty
    return float(score)


def looks_like_fundus(image: Image.Image) -> bool:
    feats = fundus_features(image)
    if feats["white_frac"] > 0.42:
        return False
    if feats["red_minus_blue"] < 5.0 and feats["corner_mean"] > 40.0:
        return False
    if feats["red_minus_green"] < -8.0:
        return False
    if fundus_score(image) >= MIN_FUNDUS_SCORE:
        return True
    if (
        feats["vignette"] >= 12.0
        and feats["std"] >= 10.0
        and feats["white_frac"] < 0.25
        and feats["corner_mean"] < 40.0
    ):
        return True
    return False


def require_fundus_like(image: Image.Image) -> None:
    if not looks_like_fundus(image):
        raise UnsuitableImageError(CLEAR_IMAGE_MESSAGE)
