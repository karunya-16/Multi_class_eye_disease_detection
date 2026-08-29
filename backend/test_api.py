"""Smoke tests for the Eye Disease Detection API.

Run from the backend folder:

    python test_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(r"D:/Practice Projects/Disease Detection")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = API_DIR.parent

TEST_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "Messidor-2+EyePac_Balanced"
    / "0"
    / "20051020_43808_0100_PP.png"
)

sys.path.insert(0, str(API_DIR))
from app import app  # noqa: E402


def fail(message: str) -> None:
    print("FAIL:", message)
    raise SystemExit(1)


def pass_(name: str, extra: str = "") -> None:
    suffix = f" | {extra}" if extra else ""
    print(f"PASS: {name}{suffix}")


def main() -> None:
    if not TEST_IMAGE.exists():
        fail(f"Test image not found: {TEST_IMAGE}")

    with TestClient(app) as client:
        root = client.get("/")
        if root.status_code != 200:
            fail(f"GET / returned {root.status_code}: {root.text}")
        root_body = root.json()
        if root_body.get("status") is not True:
            fail(f"GET / unexpected JSON: {root_body}")
        pass_("GET /", str(root_body))

        health = client.get("/health")
        if health.status_code != 200:
            fail(f"GET /health returned {health.status_code}: {health.text}")
        health_body = health.json()
        if not health_body.get("model_loaded"):
            fail(f"GET /health model not loaded: {health_body}")
        pass_("GET /health", f"model_loaded={health_body.get('model_loaded')}")

        with TEST_IMAGE.open("rb") as handle:
            predict = client.post(
                "/predict",
                files={"file": (TEST_IMAGE.name, handle, "image/png")},
            )
        if predict.status_code != 200:
            fail(f"POST /predict returned {predict.status_code}: {predict.text}")
        body = predict.json()
        if body.get("success") is not True:
            fail(f"POST /predict success is not true: {body}")
        if body.get("severity_available") is not False:
            fail(f"POST /predict must not invent severity: {body}")
        pass_(
            "POST /predict",
            f"{body['predicted_label']} confidence={body['confidence']}",
        )

        with TEST_IMAGE.open("rb") as handle:
            explain = client.post(
                "/explain",
                files={"file": (TEST_IMAGE.name, handle, "image/png")},
            )
        if explain.status_code != 200:
            fail(f"POST /explain returned {explain.status_code}: {explain.text}")
        expl = explain.json()
        if expl.get("success") is not True:
            fail(f"POST /explain success is not true: {expl}")
        if expl.get("predicted_label") != body["predicted_label"]:
            fail("POST /explain predicted class does not match /predict")
        if expl.get("confidence") != body["confidence"]:
            fail("POST /explain confidence does not match /predict")
        if expl.get("severity_available") is not False:
            fail(f"POST /explain must not invent severity: {expl}")
        if expl.get("gradcam_available") is not True:
            fail(f"POST /explain Grad-CAM unavailable: {expl.get('gradcam_error')}")
        cam = expl.get("gradcam") or {}
        if cam.get("heatmap_shape") != [7, 7]:
            fail(f"Heatmap shape before display is not (7, 7): {cam.get('heatmap_shape')}")
        if not (0.0 <= float(cam.get("heatmap_min", -1)) <= float(cam.get("heatmap_max", 2)) <= 1.0):
            fail(f"Heatmap is not normalized to [0, 1]: {cam}")
        if cam.get("predicted_label") != expl["predicted_label"]:
            fail("Grad-CAM predicted class does not match prediction")
        if not str(cam.get("heatmap_image", "")).startswith("data:image/png;base64,"):
            fail("Grad-CAM heatmap image is missing")
        pass_(
            "POST /explain",
            f"{expl['predicted_label']} heatmap={cam.get('heatmap_shape')} "
            f"range={cam.get('heatmap_min')}-{cam.get('heatmap_max')}",
        )

    print()
    print("All API smoke tests passed.")


if __name__ == "__main__":
    main()
