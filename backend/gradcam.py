"""Grad-CAM using the nested EfficientNetV2 forward-function from notebook 05.

Do not build a separate keras.Model(inputs=..., outputs=...) graph. That approach
disconnected the backbone in this project. Watch the backbone feature maps inside
GradientTape instead.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger("eye_disease_api")

TARGET_LAYER_NAME = "efficientnetv2-b0"
AUGMENTATION_LAYER_NAME = "fundus_augmentation"
GAP_LAYER_NAME = "global_average_pooling"
BN_LAYER_NAME = "classifier_bn"
DROPOUT_LAYER_NAME = "classifier_dropout"
OUTPUT_LAYER_NAME = "disease_output"

EXPECTED_HEATMAP_SHAPE = (7, 7)
EXPECTED_FEATURE_SPATIAL = (7, 7, 1280)
DISPLAY_SIZE = (224, 224)
OVERLAY_ALPHA = 0.40

REQUIRED_LAYERS = (
    AUGMENTATION_LAYER_NAME,
    TARGET_LAYER_NAME,
    GAP_LAYER_NAME,
    BN_LAYER_NAME,
    DROPOUT_LAYER_NAME,
    OUTPUT_LAYER_NAME,
)


class GradCamError(RuntimeError):
    """Raised when Grad-CAM cannot be produced for a valid prediction."""


def _jet_rgb(values: np.ndarray) -> np.ndarray:
    """Matplotlib-like jet colormap in NumPy (no extra dependency)."""
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def _pil_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _tensor_to_uint8_image(batch: tf.Tensor) -> Image.Image:
    array = np.asarray(batch[0], dtype=np.float32)
    array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise GradCamError("Could not convert the model input into an RGB image.")
    return Image.fromarray(array, mode="RGB")


def bind_gradcam_layers(model: tf.keras.Model) -> dict[str, Any]:
    missing = [name for name in REQUIRED_LAYERS if all(layer.name != name for layer in model.layers)]
    if missing:
        raise GradCamError(f"Required Grad-CAM layers were not found: {', '.join(missing)}")

    backbone = model.get_layer(TARGET_LAYER_NAME)
    output_shape = getattr(backbone, "output_shape", None)
    spatial = None
    if output_shape is not None:
        shape = output_shape[0] if isinstance(output_shape, list) else output_shape
        spatial = tuple(int(dim) if dim is not None else None for dim in shape[1:])
        if spatial != EXPECTED_FEATURE_SPATIAL:
            logger.warning(
                "Backbone output shape is %s, expected %s",
                spatial,
                EXPECTED_FEATURE_SPATIAL,
            )

    logger.info(
        "Grad-CAM bound to layer=%s output_shape=%s",
        TARGET_LAYER_NAME,
        output_shape,
    )
    return {
        "augmentation": model.get_layer(AUGMENTATION_LAYER_NAME),
        "backbone": backbone,
        "gap": model.get_layer(GAP_LAYER_NAME),
        "bn": model.get_layer(BN_LAYER_NAME),
        "dropout": model.get_layer(DROPOUT_LAYER_NAME),
        "output": model.get_layer(OUTPUT_LAYER_NAME),
        "target_layer": TARGET_LAYER_NAME,
        "feature_map_shape": list(spatial) if spatial else list(EXPECTED_FEATURE_SPATIAL),
    }


def make_gradcam_heatmap(layers: dict[str, Any], image_tensor: tf.Tensor, class_index: int) -> np.ndarray:
    """Return a (7, 7) heatmap normalized to [0, 1] for the predicted class."""
    class_index = int(class_index)

    with tf.GradientTape() as tape:
        augmented = layers["augmentation"](image_tensor, training=False)
        feature_maps = layers["backbone"](augmented, training=False)
        tape.watch(feature_maps)
        pooled = layers["gap"](feature_maps)
        normalized = layers["bn"](pooled, training=False)
        dropped = layers["dropout"](normalized, training=False)
        predictions = layers["output"](dropped)
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)
    if gradients is None:
        raise GradCamError(
            "Grad-CAM gradients are disconnected from the EfficientNetV2-B0 feature maps."
        )

    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    weighted = feature_maps[0] * pooled_gradients[0]
    heatmap = tf.reduce_sum(weighted, axis=-1)
    heatmap = tf.maximum(heatmap, 0.0)
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0.0, heatmap / max_value, heatmap)
    heatmap_np = np.asarray(heatmap, dtype=np.float32)

    if heatmap_np.ndim == 3:
        heatmap_np = heatmap_np[0]
    if heatmap_np.shape != EXPECTED_HEATMAP_SHAPE:
        raise GradCamError(
            f"Grad-CAM heatmap shape was {heatmap_np.shape}, expected {EXPECTED_HEATMAP_SHAPE}."
        )

    heatmap_np = np.clip(heatmap_np, 0.0, 1.0)
    return heatmap_np


def render_gradcam_images(batch: tf.Tensor, heatmap: np.ndarray) -> dict[str, str]:
    original = _tensor_to_uint8_image(batch).resize(DISPLAY_SIZE, Image.BILINEAR)
    heatmap_gray = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0), mode="L")
    heatmap_resized = np.asarray(
        heatmap_gray.resize(DISPLAY_SIZE, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    heatmap_color = Image.fromarray(
        np.uint8(_jet_rgb(heatmap_resized) * 255.0),
        mode="RGB",
    )
    base = np.asarray(original, dtype=np.float32)
    colored = np.asarray(heatmap_color, dtype=np.float32)
    overlay = np.clip((1.0 - OVERLAY_ALPHA) * base + OVERLAY_ALPHA * colored, 0.0, 255.0)
    overlay_image = Image.fromarray(overlay.astype(np.uint8), mode="RGB")

    return {
        "original_image": _pil_to_data_uri(original),
        "heatmap_image": _pil_to_data_uri(heatmap_color),
        "overlay_image": _pil_to_data_uri(overlay_image),
    }


def generate_gradcam_payload(
    layers: dict[str, Any],
    batch: tf.Tensor,
    predicted_class: int,
    predicted_label: str,
    confidence: float,
    confidence_percentage: float,
) -> dict[str, Any]:
    heatmap = make_gradcam_heatmap(layers, batch, predicted_class)
    images = render_gradcam_images(batch, heatmap)
    return {
        "available": True,
        "target_layer": TARGET_LAYER_NAME,
        "feature_map_shape": layers.get("feature_map_shape", list(EXPECTED_FEATURE_SPATIAL)),
        "heatmap_shape": list(heatmap.shape),
        "heatmap_min": round(float(np.min(heatmap)), 4),
        "heatmap_max": round(float(np.max(heatmap)), 4),
        "predicted_class": int(predicted_class),
        "predicted_label": predicted_label,
        "confidence": confidence,
        "confidence_percentage": confidence_percentage,
        **images,
    }
