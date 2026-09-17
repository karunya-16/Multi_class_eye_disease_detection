"""Input validation / preprocessing layer for uploaded eye photographs.

Camera-photo support lives here. It does not retrain or replace the
EfficientNetV2-B0 fundus model.
"""

from app.image_intake.constants import CLEAR_IMAGE_MESSAGE, UNSUPPORTED_FILE_MESSAGE
from app.image_intake.exceptions import UnsuitableImageError
from app.image_intake.workflow import PreparedUpload, prepare_for_inference

__all__ = [
    "CLEAR_IMAGE_MESSAGE",
    "UNSUPPORTED_FILE_MESSAGE",
    "UnsuitableImageError",
    "PreparedUpload",
    "prepare_for_inference",
]
