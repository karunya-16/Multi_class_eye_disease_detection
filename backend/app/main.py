"""Eye Disease Detection API — FastAPI backend.

Endpoints:
  GET  /health
  POST /predict
  POST /predict-with-gradcam
  GET  /history
  GET  /history/{id}
  GET  /dashboard
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.db import dashboard_stats, get_prediction, init_db, list_predictions, ping, save_prediction
from app.gradcam import build_grad_model, make_heatmap, render
from app.image_intake import UnsuitableImageError, prepare_for_inference
from app.model_service import (
    CLASS_NAMES,
    get_model,
    load_model,
    predict,
    preprocess,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eye_disease_api")

CLASS_LABELS = [CLASS_NAMES[i] for i in range(4)]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    logger.info("Model loaded successfully")
    db_info = init_db()
    logger.info("Database connected: %s", db_info)
    yield


app = FastAPI(title="Eye Disease Detection API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bad_request(msg: str):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)


def _prepare_upload(file: UploadFile, data: bytes):
    try:
        return prepare_for_inference(file.filename, file.content_type, data)
    except UnsuitableImageError as exc:
        _bad_request(str(exc))
    except ValueError as exc:
        _bad_request(str(exc))
    raise RuntimeError("Upload preparation did not return a result.")


def _parse_day(raw: str | None, *, end_of_day: bool = False) -> datetime | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        parsed = datetime.strptime(str(raw).strip(), "%Y-%m-%d")
    except ValueError:
        _bad_request("The date filter is not valid. Use YYYY-MM-DD.")
    if end_of_day:
        return parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed


def _resolve_disease(disease: str | None) -> str | None:
    value = (disease or "").strip() or None
    if value and value not in CLASS_LABELS:
        _bad_request("Unknown disease filter. Use Normal, Cataract, Diabetic Retinopathy, or Glaucoma.")
    return value


def _database_status() -> dict:
    try:
        return ping()
    except Exception:
        logger.exception("MySQL ping failed")
        return {"connected": False}


def _persist(filename: str | None, data: bytes, result: dict, gradcam: dict | None = None) -> int | None:
    try:
        return save_prediction(
            filename=filename,
            image_bytes=data,
            result=result,
            gradcam=gradcam,
        )
    except Exception:
        logger.exception("Failed to save prediction history")
        return None


@app.get("/health")
def health():
    model = get_model()
    db = _database_status()
    return {
        "status": "ok" if db.get("connected") else "degraded",
        "model_loaded": True,
        "input_shape": list(model.input_shape),
        "output_classes": 4,
        "class_labels": CLASS_LABELS,
        "database": db,
        "input_validation": {
            "enabled": True,
            "model_validated_on": "retinal_fundus",
            "camera_photos": "accepted_only_if_fundus_like_after_intake",
        },
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    prepared = _prepare_upload(file, data)
    try:
        result = predict(prepared.inference_bytes)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    history_id = _persist(file.filename, prepared.inference_bytes, result)
    payload = {"success": True, "filename": file.filename, **result}
    if history_id is not None:
        payload["history_id"] = history_id
    return payload


@app.post("/predict-with-gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    data = await file.read()
    prepared = _prepare_upload(file, data)
    try:
        result = predict(prepared.inference_bytes)
        batch = preprocess(prepared.inference_bytes)
        grad_model = build_grad_model(get_model())
        heatmap = make_heatmap(grad_model, batch, result["class_id"])
        gradcam_data = render(batch, heatmap)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Predict-with-gradcam failed")
        raise HTTPException(status_code=500, detail="Prediction or Grad-CAM failed.") from exc
    history_id = _persist(file.filename, prepared.inference_bytes, result, gradcam=gradcam_data)
    payload = {"success": True, "filename": file.filename, **result, "gradcam": gradcam_data}
    if history_id is not None:
        payload["history_id"] = history_id
    return payload


@app.get("/history")
def history(
    page: int = Query(1, ge=1),
    page_size: int = Query(8, ge=1, le=50),
    disease: str | None = Query(None),
    class_id: int | None = Query(None, ge=0, le=3),
    q: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):
    start_at = _parse_day(start_date)
    end_at = _parse_day(end_date, end_of_day=True)
    if start_at and end_at and start_at > end_at:
        _bad_request("The date range is not valid.")
    try:
        listing = list_predictions(
            disease=_resolve_disease(disease),
            class_id=class_id,
            query=q,
            page=page,
            page_size=page_size,
            start_at=start_at,
            end_at=end_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("History list failed")
        raise HTTPException(status_code=500, detail="History could not be loaded.") from exc
    return {"success": True, **listing}


@app.get("/history/{record_id}")
def history_detail(record_id: int):
    try:
        record = get_prediction(record_id)
    except Exception as exc:
        logger.exception("History detail failed")
        raise HTTPException(status_code=500, detail="History record could not be loaded.") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="History record not found.")
    return {"success": True, **record}


@app.get("/dashboard")
def dashboard(
    disease: str | None = Query(None),
    class_id: int | None = Query(None, ge=0, le=3),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):
    start_at = _parse_day(start_date)
    end_at = _parse_day(end_date, end_of_day=True)
    if start_at and end_at and start_at > end_at:
        _bad_request("The date range is not valid.")
    try:
        stats = dashboard_stats(
            disease=_resolve_disease(disease),
            class_id=class_id,
            start_at=start_at,
            end_at=end_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Dashboard stats failed")
        raise HTTPException(status_code=500, detail="Dashboard could not be loaded.") from exc
    return {"success": True, **stats}


@app.exception_handler(HTTPException)
async def http_exc_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "error": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exc_handler(_, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"success": False, "error": "Internal server error."})
