"""End-to-end tests for the 4-class Eye Disease Detection API.

Prefers the running server at http://127.0.0.1:8000 so the live product
is tested. Falls back to TestClient if the server is not up.

    python test_api.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

from test_support.image_cases import (
    blurry_fundus_jpeg,
    external_eye_jpeg,
    fundus_camera_style_jpeg,
    fundus_webp,
    unsupported_scene_jpeg,
)

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
LIVE_BASE = "http://127.0.0.1:8000"

CLASS_NAMES = ["Normal", "Cataract", "Diabetic Retinopathy", "Glaucoma"]
CLASS_IMAGES = {
    "Normal": PROJECT_ROOT
    / "dataset"
    / "Eye Disease detection"
    / "normal"
    / "3442_right.jpg",
    "Cataract": PROJECT_ROOT
    / "dataset"
    / "Eye Disease detection"
    / "cataract"
    / "_114_5711178.jpg",
    "Diabetic Retinopathy": PROJECT_ROOT
    / "dataset"
    / "Eye Disease detection"
    / "diabetic_retinopathy"
    / "10761_right.jpeg",
    "Glaucoma": PROJECT_ROOT
    / "dataset"
    / "Eye Disease detection"
    / "glaucoma"
    / "_0_4517448.jpg",
}


def fail(message: str) -> None:
    print("FAIL:", message)
    raise SystemExit(1)


def pass_(name: str, extra: str = "") -> None:
    suffix = f" | {extra}" if extra else ""
    print(f"PASS: {name}{suffix}")


def _assert_prediction_payload(body: dict, *, context: str, expected: str | None = None) -> None:
    if body.get("success") is not True:
        fail(f"{context}: success is not true: {body}")
    if body.get("disease") not in CLASS_NAMES:
        fail(f"{context}: unexpected disease: {body.get('disease')}")
    class_id = body.get("class_id")
    if class_id not in (0, 1, 2, 3):
        fail(f"{context}: class_id must be 0–3: {class_id}")
    if CLASS_NAMES[int(class_id)] != body["disease"]:
        fail(f"{context}: class_id does not match disease")
    if expected and body["disease"] != expected:
        fail(f"{context}: expected {expected}, got {body.get('disease')}")
    if not isinstance(body.get("confidence"), (int, float)):
        fail(f"{context}: missing confidence")
    probs = body.get("probabilities")
    if not isinstance(probs, dict):
        fail(f"{context}: missing probabilities")
    for name in CLASS_NAMES:
        value = probs.get(name)
        if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0 + 1e-6):
            fail(f"{context}: invalid probability for {name}: {value}")


def _mime(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def _open_live() -> httpx.Client:
    client = httpx.Client(base_url=LIVE_BASE, timeout=180.0)
    try:
        health = client.get("/health")
    except httpx.HTTPError as exc:
        client.close()
        fail(f"Live API is not reachable at {LIVE_BASE}: {exc}")
    if health.status_code != 200:
        client.close()
        fail(f"GET /health returned {health.status_code}: {health.text}")
    return client


def main() -> None:
    for name, path in CLASS_IMAGES.items():
        if not path.exists():
            fail(f"Test image for {name} not found: {path}")

    client = _open_live()
    saved_ids: dict[str, int] = {}
    try:
        health_body = client.get("/health").json()
        if health_body.get("model_loaded") is not True:
            fail(f"GET /health unexpected JSON: {health_body}")
        if health_body.get("output_classes") != 4 or health_body.get("class_labels") != CLASS_NAMES:
            fail(f"GET /health class contract mismatch: {health_body}")
        db = health_body.get("database") or {}
        if db.get("connected") is not True or db.get("table") != "prediction_history":
            fail(f"MySQL is not ready: {db}")
        pass_("GET /health", f"db={db.get('database')} table={db.get('table')}")
        print(json.dumps(health_body, indent=2))
        print()

        before_total = int((client.get("/dashboard").json() or {}).get("total_predictions") or 0)

        cataract = CLASS_IMAGES["Cataract"]
        with cataract.open("rb") as handle:
            predict = client.post(
                "/predict",
                files={"file": (cataract.name, handle, _mime(cataract))},
            )
        if predict.status_code != 200:
            fail(f"POST /predict returned {predict.status_code}: {predict.text}")
        predict_body = predict.json()
        _assert_prediction_payload(predict_body, context="POST /predict", expected="Cataract")
        predict_id = predict_body.get("history_id")
        if not isinstance(predict_id, int) or predict_id <= 0:
            fail(f"POST /predict did not save to MySQL: {predict_body}")
        pass_(
            "POST /predict",
            f"{predict_body['disease']} confidence={predict_body['confidence']} history_id={predict_id}",
        )
        print(json.dumps({k: predict_body[k] for k in ("disease", "class_id", "confidence", "probabilities", "history_id")}, indent=2))
        print()

        for expected, path in CLASS_IMAGES.items():
            with path.open("rb") as handle:
                response = client.post(
                    "/predict-with-gradcam",
                    files={"file": (path.name, handle, _mime(path))},
                )
            if response.status_code != 200:
                fail(f"POST /predict-with-gradcam {expected} returned {response.status_code}: {response.text}")
            body = response.json()
            _assert_prediction_payload(body, context=f"POST /predict-with-gradcam {expected}", expected=expected)
            cam = body.get("gradcam") or {}
            if cam.get("target_layer") != "top_activation" or cam.get("heatmap_shape") != [7, 7]:
                fail(f"Grad-CAM metadata invalid for {expected}: {cam}")
            for key in ("original_image", "heatmap_image", "overlay_image"):
                if not str(cam.get(key, "")).startswith("data:image/png;base64,"):
                    fail(f"Grad-CAM {key} missing for {expected}")
            history_id = body.get("history_id")
            if not isinstance(history_id, int) or history_id <= 0:
                fail(f"{expected} was not saved to MySQL")
            saved_ids[expected] = history_id
            pass_(
                f"4-class {expected}",
                f"class_id={body['class_id']} confidence={body['confidence']} history_id={history_id}",
            )
        print()

        bad = client.post("/predict", files={"file": ("notes.txt", b"not-an-image", "text/plain")})
        if bad.status_code != 400:
            fail(f"POST /predict invalid file returned {bad.status_code}: {bad.text}")
        bad_body = bad.json()
        if bad_body.get("success") is not False or not bad_body.get("error"):
            fail(f"POST /predict invalid file unexpected JSON: {bad_body}")
        pass_("Invalid file rejected", json.dumps(bad_body))

        for label, filename, payload in (
            ("blurry", "blurry.jpg", blurry_fundus_jpeg()),
            ("unsupported scene", "park.jpg", unsupported_scene_jpeg()),
            ("external-eye camera photo", "selfie_eye.jpg", external_eye_jpeg()),
        ):
            response = client.post(
                "/predict",
                files={"file": (filename, payload, "image/jpeg")},
            )
            if response.status_code != 400:
                fail(f"{label} returned {response.status_code}: {response.text}")
            body = response.json()
            if body.get("error") != "Please upload a clear image":
                fail(f"{label} unexpected JSON: {body}")
            pass_(f"{label} rejected", body.get("error"))

        missing = client.get("/history/0")
        if missing.status_code != 404:
            fail(f"GET /history/0 returned {missing.status_code}: {missing.text}")
        pass_("GET /history/0", "404 as expected")
        print()

        history = client.get("/history", params={"page": 1, "page_size": 20})
        if history.status_code != 200:
            fail(f"GET /history returned {history.status_code}: {history.text}")
        history_body = history.json()
        ids = [item.get("id") for item in history_body.get("items") or []]
        for disease, record_id in saved_ids.items():
            if record_id not in ids:
                fail(f"GET /history missing {disease} id={record_id}")
        newest = history_body["items"][0]
        if newest.get("id") != saved_ids["Glaucoma"]:
            fail(f"GET /history is not newest-first: {newest}")
        if newest.get("gradcam_available") is not True or not newest.get("gradcam_path"):
            fail(f"GET /history Grad-CAM path missing: {newest}")
        pass_("GET /history", f"total={history_body.get('total')} newest={newest.get('disease')}")

        for disease, record_id in saved_ids.items():
            detail = client.get(f"/history/{record_id}")
            if detail.status_code != 200:
                fail(f"GET /history/{record_id} returned {detail.status_code}: {detail.text}")
            body = detail.json()
            if body.get("disease") != disease:
                fail(f"GET /history/{record_id} disease mismatch: {body.get('disease')}")
            if not str(body.get("uploaded_image") or "").startswith("data:image"):
                fail(f"GET /history/{record_id} uploaded image missing")
            if not str((body.get("gradcam") or {}).get("heatmap_image") or "").startswith("data:image"):
                fail(f"GET /history/{record_id} Grad-CAM missing")
            pass_(f"GET /history/{record_id}", f"disease={disease}")
        print()

        camera = fundus_camera_style_jpeg()
        cam_response = client.post(
            "/predict-with-gradcam",
            files={"file": ("phone_cataract.jpg", camera, "image/jpeg")},
        )
        if cam_response.status_code != 200:
            fail(f"camera-style image returned {cam_response.status_code}: {cam_response.text}")
        cam_body = cam_response.json()
        _assert_prediction_payload(cam_body, context="camera-style", expected="Cataract")
        if (cam_body.get("gradcam") or {}).get("target_layer") != "top_activation":
            fail(f"camera-style Grad-CAM layer invalid: {cam_body.get('gradcam')}")
        pass_("camera-style eye photograph", f"{cam_body['disease']} history_id={cam_body.get('history_id')}")

        webp = fundus_webp()
        webp_response = client.post(
            "/predict",
            files={"file": ("cataract.webp", webp, "image/webp")},
        )
        if webp_response.status_code != 200:
            fail(f"webp fundus returned {webp_response.status_code}: {webp_response.text}")
        webp_body = webp_response.json()
        _assert_prediction_payload(webp_body, context="webp fundus", expected="Cataract")
        pass_("webp fundus photograph", f"{webp_body['disease']} history_id={webp_body.get('history_id')}")
        print()

        dash = client.get("/dashboard")
        if dash.status_code != 200:
            fail(f"GET /dashboard returned {dash.status_code}: {dash.text}")
        dash_body = dash.json()
        if dash_body.get("success") is not True:
            fail(f"GET /dashboard unexpected JSON: {dash_body}")
        if int(dash_body.get("total_predictions") or 0) < before_total + 7:
            fail(f"Dashboard did not persist new predictions: {dash_body}")
        distribution = {row.get("disease"): int(row.get("count") or 0) for row in dash_body.get("class_distribution") or []}
        if set(distribution) != set(CLASS_NAMES):
            fail(f"Dashboard class_distribution is not 4 named diseases: {distribution}")
        for name in CLASS_NAMES:
            if distribution.get(name, 0) < 1:
                fail(f"Dashboard has no {name} rows: {distribution}")
        pass_(
            "GET /dashboard",
            f"total={dash_body.get('total_predictions')} "
            f"avg_confidence={dash_body.get('average_confidence')} "
            f"counts={distribution}",
        )
        print(json.dumps({"total_predictions": dash_body.get("total_predictions"), "class_distribution": dash_body.get("class_distribution")}, indent=2))
    finally:
        client.close()

    print()
    print("All 4-class API end-to-end tests passed.")


if __name__ == "__main__":
    main()
