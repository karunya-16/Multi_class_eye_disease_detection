"""Accepted upload formats for camera and gallery photographs."""

from __future__ import annotations

from pathlib import Path

from app.image_intake.constants import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    EMPTY_UPLOAD_MESSAGE,
    UNSUPPORTED_FILE_MESSAGE,
)


def validate_file_format(filename: str | None, content_type: str | None, data: bytes) -> None:
    if not filename or not data:
        raise ValueError(EMPTY_UPLOAD_MESSAGE)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(UNSUPPORTED_FILE_MESSAGE)
    if content_type:
        ctype = content_type.split(";")[0].strip().lower()
        if ctype and ctype not in ALLOWED_CONTENT_TYPES:
            raise ValueError(UNSUPPORTED_FILE_MESSAGE)
