"""Grad-CAM for the 4-class EfficientNetV2-B0 model (flat architecture).

Target layer: top_activation (7×7×1280), verified in notebook 07_gradcam_v2.
Uses tf.keras.Model sub-graph approach — the only method that keeps gradients
connected in this flat (non-nested) saved model.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import tensorflow as tf
from PIL import Image

GRADCAM_LAYER = "top_activation"
OUTPUT_LAYER = "disease_output"
DISPLAY_SIZE = (224, 224)
OVERLAY_ALPHA = 0.40

_grad_model: tf.keras.Model | None = None


def build_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    global _grad_model
    if _grad_model is None:
        target = model.get_layer(GRADCAM_LAYER)
        output = model.get_layer(OUTPUT_LAYER)
        _grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target.output, output.output],
        )
    return _grad_model


def make_heatmap(grad_model: tf.keras.Model, batch: np.ndarray, class_idx: int) -> np.ndarray:
    tensor = tf.cast(batch, tf.float32)
    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(tensor, training=False)
        tape.watch(feature_maps)
        class_score = predictions[:, class_idx]
    grads = tape.gradient(class_score, feature_maps)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients are None")
    pooled = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(feature_maps * pooled[:, tf.newaxis, tf.newaxis, :], axis=-1)
    heatmap = tf.nn.relu(heatmap)[0].numpy()
    max_val = heatmap.max()
    if max_val > 1e-8:
        heatmap = heatmap / max_val
    return heatmap.astype(np.float32)


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render(batch: np.ndarray, heatmap: np.ndarray) -> dict:
    original = Image.fromarray(
        np.clip(batch[0], 0, 255).astype(np.uint8), mode="RGB"
    ).resize(DISPLAY_SIZE, Image.Resampling.BILINEAR)

    heatmap_resized = np.asarray(
        Image.fromarray(np.uint8(heatmap * 255), mode="L").resize(DISPLAY_SIZE, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    colored = Image.fromarray(np.uint8(_jet_colormap(heatmap_resized) * 255), mode="RGB")

    overlay_arr = np.clip(
        (1 - OVERLAY_ALPHA) * np.asarray(original, np.float32)
        + OVERLAY_ALPHA * np.asarray(colored, np.float32),
        0, 255,
    ).astype(np.uint8)
    overlay = Image.fromarray(overlay_arr, mode="RGB")

    return {
        "original_image": _to_png_b64(original),
        "heatmap_image": _to_png_b64(colored),
        "overlay_image": _to_png_b64(overlay),
        "heatmap_shape": list(heatmap.shape),
        "heatmap_min": round(float(heatmap.min()), 4),
        "heatmap_max": round(float(heatmap.max()), 4),
        "target_layer": GRADCAM_LAYER,
    }
