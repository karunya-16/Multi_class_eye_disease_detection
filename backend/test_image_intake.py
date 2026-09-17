"""Unit tests for the image intake layer (no model weights required)."""

from __future__ import annotations

from pathlib import Path

from app.image_intake import CLEAR_IMAGE_MESSAGE, UnsuitableImageError, prepare_for_inference
from test_support.image_cases import (
    blurry_fundus_jpeg,
    CATARACT,
    external_eye_jpeg,
    fundus_camera_style_jpeg,
    fundus_webp,
    unsupported_scene_jpeg,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_IMAGES = {
    "Normal": PROJECT_ROOT / "dataset" / "Eye Disease detection" / "normal" / "3442_right.jpg",
    "Cataract": CATARACT,
    "Diabetic Retinopathy": PROJECT_ROOT
    / "dataset"
    / "Eye Disease detection"
    / "diabetic_retinopathy"
    / "10761_right.jpeg",
    "Glaucoma": PROJECT_ROOT / "dataset" / "Eye Disease detection" / "glaucoma" / "_0_4517448.jpg",
}


def fail(message: str) -> None:
    print("FAIL:", message)
    raise SystemExit(1)


def pass_(name: str, extra: str = "") -> None:
    suffix = f" | {extra}" if extra else ""
    print(f"PASS: {name}{suffix}")


def main() -> None:
    for name, path in CLASS_IMAGES.items():
        if not path.exists():
            fail(f"Missing {name} image: {path}")
        prepared = prepare_for_inference(path.name, "image/jpeg", path.read_bytes())
        if prepared.source != "original":
            fail(f"{name} should keep original bytes, got source={prepared.source}")
        pass_(f"intake retinal {name}", f"source={prepared.source} {prepared.width}x{prepared.height}")

    camera = fundus_camera_style_jpeg()
    prepared = prepare_for_inference("phone_cataract.jpg", "image/jpeg", camera)
    if prepared.source != "camera_roi":
        fail(f"camera-style image should be cropped, got source={prepared.source}")
    pass_("intake camera-style fundus", f"source={prepared.source} {prepared.width}x{prepared.height}")

    webp = fundus_webp()
    prepared = prepare_for_inference("cataract.webp", "image/webp", webp)
    pass_("intake webp fundus", f"source={prepared.source}")

    try:
        prepare_for_inference("blurry.jpg", "image/jpeg", blurry_fundus_jpeg())
        fail("blurry fundus should be rejected")
    except UnsuitableImageError as exc:
        if str(exc) != CLEAR_IMAGE_MESSAGE:
            fail(f"blurry message mismatch: {exc}")
        pass_("intake blurry rejected", CLEAR_IMAGE_MESSAGE)

    try:
        prepare_for_inference("park.jpg", "image/jpeg", unsupported_scene_jpeg())
        fail("unsupported scene should be rejected")
    except UnsuitableImageError as exc:
        if str(exc) != CLEAR_IMAGE_MESSAGE:
            fail(f"unsupported scene message mismatch: {exc}")
        pass_("intake unsupported scene rejected", CLEAR_IMAGE_MESSAGE)

    try:
        prepare_for_inference("selfie.jpg", "image/jpeg", external_eye_jpeg())
        fail("external-eye photo should be rejected")
    except UnsuitableImageError as exc:
        if str(exc) != CLEAR_IMAGE_MESSAGE:
            fail(f"external-eye message mismatch: {exc}")
        pass_("intake external-eye rejected", CLEAR_IMAGE_MESSAGE)

    try:
        prepare_for_inference("notes.txt", "text/plain", b"not-an-image")
        fail("invalid file type should be rejected")
    except ValueError as exc:
        if "Unsupported file type" not in str(exc):
            fail(f"invalid file message mismatch: {exc}")
        pass_("intake invalid file type rejected", str(exc))

    print()
    print("Image intake unit tests passed.")


if __name__ == "__main__":
    main()
