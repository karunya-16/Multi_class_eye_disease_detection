"""Raised when an upload must not be sent to the retinal model."""

from __future__ import annotations


class UnsuitableImageError(ValueError):
    """The file is an image, but it is not a usable retinal/fundus input."""
