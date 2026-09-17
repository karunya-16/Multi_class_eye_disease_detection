"""Model loading and prediction service for the 4-class eye disease model."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "model" / "efficientnetv2_b0_4class_best.keras"

IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "Normal", 1: "Cataract", 2: "Diabetic Retinopathy", 3: "Glaucoma"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/x-png", "application/octet-stream"}

_model: keras.Model | None = None


def load_model() -> keras.Model:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = keras.models.load_model(MODEL_PATH)
        assert int(_model.output_shape[-1]) == 4, "Expected 4-class model"
    return _model


def get_model() -> keras.Model:
    if _model is None:
        raise RuntimeError("Model not loaded")
    return _model


def validate_upload(filename: str | None, content_type: str | None, data: bytes) -> None:
    if not filename or not data:
        raise ValueError("Empty upload")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Use JPG, JPEG, or PNG.")
    if content_type:
        ctype = content_type.split(";")[0].strip().lower()
        if ctype not in ALLOWED_CONTENT_TYPES:
            raise ValueError("Unsupported content type.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Invalid image file.") from exc


def preprocess(data: bytes) -> np.ndarray:
    """PIL RGB → resize 224×224 BILINEAR → float32 → batch. No /255."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(data: bytes) -> dict:
    batch = preprocess(data)
    probs = get_model().predict(batch, verbose=0)[0]
    class_id = int(np.argmax(probs))
    return {
        "disease": CLASS_NAMES[class_id],
        "class_id": class_id,
        "confidence": round(float(probs[class_id]), 4),
        "probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(4)},
    }
