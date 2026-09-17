"""Orchestrate intake: format → decode → quality → fundus check → optional ROI.

If this workflow accepts the upload, the returned bytes are fed to the
existing ``app.model_service.predict`` / ``preprocess`` functions unchanged.
Verified retinal JPEGs that need no orientation fix or crop are passed
through as the original file bytes so inference matches the 84.94% pipeline.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image

from app.image_intake.constants import CROP_AREA_RATIO
from app.image_intake.decode import decode_rgb, orientation_was_corrected
from app.image_intake.formats import validate_file_format
from app.image_intake.fundus_cues import require_fundus_like
from app.image_intake.quality import check_quality
from app.image_intake.roi import select_best_roi


@dataclass(frozen=True)
class PreparedUpload:
    """Bytes ready for ``model_service.predict`` / ``preprocess``."""

    inference_bytes: bytes
    source: str
    width: int
    height: int


def _to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _area(image: Image.Image) -> int:
    return image.size[0] * image.size[1]


def prepare_for_inference(filename: str | None, content_type: str | None, data: bytes) -> PreparedUpload:
    """Validate and, when needed, prepare camera photographs for the fundus model."""
    validate_file_format(filename, content_type, data)
    rgb = decode_rgb(data)
    check_quality(rgb)

    oriented = orientation_was_corrected(data, rgb)
    roi = select_best_roi(rgb)
    cropped = _area(roi) < CROP_AREA_RATIO * _area(rgb)

    if cropped:
        check_quality(roi)
        require_fundus_like(roi)
        return PreparedUpload(
            inference_bytes=_to_png_bytes(roi),
            source="camera_roi",
            width=roi.size[0],
            height=roi.size[1],
        )

    require_fundus_like(rgb)
    if oriented:
        return PreparedUpload(
            inference_bytes=_to_png_bytes(rgb),
            source="orientation_corrected",
            width=rgb.size[0],
            height=rgb.size[1],
        )

    return PreparedUpload(
        inference_bytes=data,
        source="original",
        width=rgb.size[0],
        height=rgb.size[1],
    )
