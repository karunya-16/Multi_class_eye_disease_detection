"""Thresholds and user-facing messages for the image intake layer.

This layer does not change the EfficientNetV2-B0 weights, the verified
PIL RGB → 224×224 BILINEAR → float32 0–255 inference preprocess, or Grad-CAM.
"""

from __future__ import annotations

CLEAR_IMAGE_MESSAGE = "Please upload a clear image"

UNSUPPORTED_FILE_MESSAGE = (
    "Unsupported file type. Use JPG, JPEG, PNG, WEBP, BMP, TIFF, or HEIC."
)
EMPTY_UPLOAD_MESSAGE = "Empty upload"
INVALID_IMAGE_MESSAGE = "Invalid image file."

# Camera / gallery formats accepted before the existing inference pipeline.
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/x-png",
    "image/webp",
    "image/bmp",
    "image/x-ms-bmp",
    "image/tiff",
    "image/tif",
    "image/heic",
    "image/heif",
    "application/octet-stream",
}

MIN_SIDE_PX = 64
MIN_MEAN_BRIGHTNESS = 8.0
MAX_MEAN_BRIGHTNESS = 242.0
MIN_HIGHFREQ = 0.28
MIN_STRUCTURE_STD = 8.0
MIN_FUNDUS_SCORE = 0.22
CROP_AREA_RATIO = 0.90
