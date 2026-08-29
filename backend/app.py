"""Eye Disease Detection API.

Loads the saved EfficientNetV2-B0 checkpoint once at startup and serves
single-image predictions. The model is never retrained or overwritten.

This is the product backend. Preprocessing matches
notebooks/04_efficientnetv2_training.ipynb:
decode RGB, bilinear resize to 224x224, keep float32 pixels in 0-255.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

from gradcam import GradCamError, bind_gradcam_layers, generate_gradcam_payload
from history_store import (
    REQUEST_ID_RE,
    get_analysis,
    get_dashboard_analytics,
    get_image_bytes,
    init_history_store,
    list_analyses,
    save_analysis,
)

logger = logging.getLogger("eye_disease_api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

HARDCODED_ROOT = Path(r"D:/Practice Projects/Disease Detection")
API_DIR = Path(__file__).resolve().parent

if HARDCODED_ROOT.exists():
    PROJECT_ROOT = HARDCODED_ROOT
else:
    PROJECT_ROOT = API_DIR.parent

MODEL_PATH = PROJECT_ROOT / "model" / "efficientnetv2_finetuned_best.keras"

IMG_SIZE = (224, 224)
NUM_CLASSES = 5
CLASS_NAMES = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/x-png",
    "application/octet-stream",
}

PREDICTION_MESSAGE = (
    "AI screening prediction only. Confidence is not disease severity "
    "or percentage of eye damage."
)

MEDICAL_DISCLAIMER = (
    "This system is an AI-based research prototype for eye disease screening. "
    "The prediction and confidence score are not a medical diagnosis and should "
    "not be interpreted as disease severity or percentage of eye damage. "
    "Please consult a qualified eye-care professional for clinical evaluation."
)

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app_state: dict[str, Any] = {
    "model": None,
    "model_loaded": False,
    "model_path": str(MODEL_PATH),
    "gradcam_layers": None,
    "gradcam_ready": False,
    "gradcam_target_layer": "efficientnetv2-b0",
}


def resolve_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    matches = list((PROJECT_ROOT / "model").rglob("*finetuned*best*.keras"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Fine-tuned checkpoint not found: {MODEL_PATH}"
    )


def load_model_once() -> keras.Model:
    path = resolve_model_path()
    logger.info("Loading model from %s", path)
    model = keras.models.load_model(path)
    app_state["model"] = model
    app_state["model_loaded"] = True
    app_state["model_path"] = str(path)
    logger.info(
        "Model loaded: name=%s input=%s output=%s",
        getattr(model, "name", "unknown"),
        model.input_shape,
        model.output_shape,
    )
    try:
        app_state["gradcam_layers"] = bind_gradcam_layers(model)
        app_state["gradcam_ready"] = True
    except Exception:
        logger.exception("Grad-CAM layers could not be bound; /predict remains available")
        app_state["gradcam_layers"] = None
        app_state["gradcam_ready"] = False
    return model


def get_model() -> keras.Model:
    model = app_state.get("model")
    if model is None:
        raise RuntimeError("Model is not loaded.")
    return model


def error_payload(message: str) -> dict[str, Any]:
    return {"success": False, "error": message}


def parse_optional_date(raw: str | None, *, end_of_day: bool = False) -> datetime | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The date filter is not valid. Use YYYY-MM-DD.",
        )
    if end_of_day:
        return parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed


def resolve_date_range(
    start_date: str | None,
    end_date: str | None,
) -> tuple[datetime | None, datetime | None]:
    start_at = parse_optional_date(start_date, end_of_day=False)
    end_at = parse_optional_date(end_date, end_of_day=True)
    if start_at and end_at and start_at > end_at:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The date range is not valid.",
        )
    return start_at, end_at


def resolve_class_filter(
    predicted_class: int | None,
    predicted_label: str | None,
) -> int | None:
    label = predicted_label.strip() if isinstance(predicted_label, str) and predicted_label.strip() else None
    if label and label not in CLASS_NAMES.values():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown predicted class filter. Use Class 0 through Class 4.",
        )
    if predicted_class is not None:
        return int(predicted_class)
    if label:
        return next(index for index, name in CLASS_NAMES.items() if name == label)
    return None


def validate_upload(filename: str | None, content_type: str | None, contents: bytes) -> str:
    if not filename or not str(filename).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty upload. Please attach an image file in the 'file' field.",
        )

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty upload. The attached file has no content.",
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Upload a JPG, JPEG, or PNG image.",
        )

    if content_type:
        ctype = content_type.split(";")[0].strip().lower()
        if ctype not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Upload a JPG, JPEG, or PNG image.",
            )

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image. The file could not be read as a JPG/JPEG/PNG.",
        )

    try:
        with Image.open(io.BytesIO(contents)) as image:
            fmt = (image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image. The file could not be read as a JPG/JPEG/PNG.",
        )

    if fmt not in {"JPEG", "JPG", "PNG"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Upload a JPG, JPEG, or PNG image.",
        )

    return Path(filename).name


def preprocess_image_bytes(contents: bytes) -> tf.Tensor:
    """Match training load_image(): decode RGB, resize 224x224, keep 0-255 float32."""
    try:
        image = tf.image.decode_image(
            contents,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            image,
            IMG_SIZE,
            method=tf.image.ResizeMethod.BILINEAR,
        )
        image = tf.cast(image, tf.float32)
    except (tf.errors.InvalidArgumentError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image. The file could not be decoded for prediction.",
        ) from exc

    pixel_min = float(tf.reduce_min(image))
    pixel_max = float(tf.reduce_max(image))
    if pixel_min < 0.0 or pixel_max > 255.0 + 1e-3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image. Pixel values are outside the expected 0-255 range.",
        )

    return tf.expand_dims(image, axis=0)


def run_prediction(batch: tf.Tensor) -> dict[str, Any]:
    model = get_model()
    try:
        probabilities = model.predict(batch, verbose=0)[0]
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction failed. Please try again.",
        ) from exc

    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (NUM_CLASSES,):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction failed. Unexpected output shape.",
        )

    class_id = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    class_probabilities = {
        CLASS_NAMES[i]: round(float(probabilities[i]), 4)
        for i in range(NUM_CLASSES)
    }

    return {
        "predicted_class": class_id,
        "predicted_label": CLASS_NAMES[class_id],
        "confidence": round(confidence, 4),
        "confidence_percentage": round(confidence * 100.0, 2),
        "class_probabilities": class_probabilities,
    }


def persist_history(
    *,
    request_id: str,
    filename: str,
    contents: bytes,
    prediction: dict[str, Any],
    original_bytes: bytes | None = None,
    heatmap_bytes: bytes | None = None,
    overlay_bytes: bytes | None = None,
    heatmap_shape: Any = None,
    target_layer: str | None = None,
) -> int | None:
    return save_analysis(
        request_id=request_id,
        filename=filename,
        uploaded_bytes=contents,
        predicted_class=prediction["predicted_class"],
        predicted_label=prediction["predicted_label"],
        confidence=prediction["confidence"],
        confidence_percentage=prediction["confidence_percentage"],
        class_probabilities=prediction["class_probabilities"],
        original_bytes=original_bytes,
        heatmap_bytes=heatmap_bytes,
        overlay_bytes=overlay_bytes,
        heatmap_shape=heatmap_shape,
        target_layer=target_layer,
    )


async def read_optional_image(upload: UploadFile | None) -> bytes | None:
    if upload is None:
        return None
    filename = (upload.filename or "").strip()
    if not filename:
        return None
    data = await upload.read()
    return data or None


def validate_history_prediction(
    predicted_class: int,
    predicted_label: str,
    confidence: float,
    confidence_percentage: float,
    class_probabilities: dict[str, float],
) -> dict[str, Any]:
    if predicted_class not in CLASS_NAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The analysis result could not be saved because the predicted class is invalid.",
        )
    expected_label = CLASS_NAMES[predicted_class]
    if predicted_label != expected_label:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The analysis result could not be saved because the predicted label does not match the class.",
        )
    for name in CLASS_NAMES.values():
        value = class_probabilities.get(name)
        if not isinstance(value, (int, float)) or value != value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The analysis result could not be saved because class probabilities are incomplete.",
            )
        if float(value) < -1e-6 or float(value) > 1.0 + 1e-6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The analysis result could not be saved because a class probability is out of range.",
            )
    max_prob = max(float(class_probabilities[CLASS_NAMES[i]]) for i in range(NUM_CLASSES))
    if abs(float(confidence) - max_prob) > 0.02:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The analysis result could not be saved because confidence does not match the class probabilities.",
        )
    expected_pct = round(float(confidence) * 100.0, 2)
    if abs(float(confidence_percentage) - expected_pct) > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The analysis result could not be saved because confidence percentage is invalid.",
        )
    return {
        "predicted_class": predicted_class,
        "predicted_label": expected_label,
        "confidence": round(float(confidence), 4),
        "confidence_percentage": round(float(confidence_percentage), 2),
        "class_probabilities": {
            CLASS_NAMES[i]: round(float(class_probabilities[CLASS_NAMES[i]]), 4)
            for i in range(NUM_CLASSES)
        },
    }


def parse_heatmap_shape(raw: str | None) -> Any:
    if not raw or not str(raw).strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed


def history_image_url(request: Request, analysis_id: int, kind: str) -> str:
    return str(request.base_url).rstrip("/") + f"/history/{analysis_id}/image/{kind}"


def serialize_history_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "created_at": record["created_at"],
        "filename": record["filename"],
        "predicted_class": record["predicted_class"],
        "predicted_label": record["predicted_label"],
        "confidence": record["confidence"],
        "confidence_percentage": record["confidence_percentage"],
        "severity_available": False,
        "gradcam_available": record["gradcam_available"],
    }


def serialize_history_detail(request: Request, record: dict[str, Any]) -> dict[str, Any]:
    analysis_id = record["id"]
    uploaded_url = (
        history_image_url(request, analysis_id, "uploaded")
        if record.get("has_uploaded")
        else None
    )
    gradcam = None
    if record["gradcam_available"]:
        gradcam = {
            "available": True,
            "target_layer": record.get("target_layer"),
            "heatmap_shape": record.get("heatmap_shape"),
            "predicted_class": record["predicted_class"],
            "predicted_label": record["predicted_label"],
            "confidence": record["confidence"],
            "confidence_percentage": record["confidence_percentage"],
            "original_image": history_image_url(request, analysis_id, "original"),
            "heatmap_image": history_image_url(request, analysis_id, "heatmap"),
            "overlay_image": history_image_url(request, analysis_id, "overlay"),
        }
    return {
        "success": True,
        "id": analysis_id,
        "created_at": record["created_at"],
        "filename": record["filename"],
        "predicted_class": record["predicted_class"],
        "predicted_label": record["predicted_label"],
        "confidence": record["confidence"],
        "confidence_percentage": record["confidence_percentage"],
        "class_probabilities": record["class_probabilities"],
        "severity_available": False,
        "message": PREDICTION_MESSAGE,
        "uploaded_image": uploaded_url,
        "gradcam_available": bool(record["gradcam_available"]),
        "gradcam": gradcam,
        "gradcam_error": None if record["gradcam_available"] else "Grad-CAM was not stored for this analysis.",
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        init_history_store()
    except Exception:
        logger.exception("MySQL history store is unavailable; prediction remains available")
    try:
        load_model_once()
    except Exception:
        logger.exception("Failed to load model at startup")
        app_state["model"] = None
        app_state["model_loaded"] = False
        raise
    yield
    app_state["model"] = None
    app_state["model_loaded"] = False


app = FastAPI(
    title="Eye Disease Detection API",
    description=(
        "Research prototype API for 5-class retinal fundus screening "
        "with EfficientNetV2-B0.\n\n"
        f"**Disclaimer:** {MEDICAL_DISCLAIMER}"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    detail = exc.detail
    if not isinstance(detail, str):
        detail = str(detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(detail),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    path = request.url.path.rstrip("/")
    if "/history" in path or "/analytics" in path:
        message = "The request was not valid."
    else:
        message = "Invalid or empty upload. Attach a JPG/JPEG/PNG file in the 'file' field."
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_payload(message),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception):
    logger.exception("Unhandled API error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_payload("Internal server error."),
    )


@app.get("/", tags=["system"])
def root() -> dict[str, Any]:
    return {
        "status": True,
        "message": "Eye Disease Detection API is running",
    }


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    model_loaded = bool(app_state.get("model_loaded") and app_state.get("model") is not None)
    payload = {
        "status": model_loaded,
        "api": "ok",
        "model_loaded": model_loaded,
        "model_path": app_state.get("model_path"),
        "input_size": list(IMG_SIZE),
        "num_classes": NUM_CLASSES,
        "class_labels": [CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        "severity_available": False,
        "gradcam_available": bool(app_state.get("gradcam_ready")),
        "gradcam_target_layer": app_state.get("gradcam_target_layer"),
        "message": PREDICTION_MESSAGE,
    }
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API is running but the model is not loaded.",
        )
    return payload


@app.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(..., description="Fundus / eye image (JPG, JPEG, or PNG)")) -> dict[str, Any]:
    """Upload a fundus image and return a 5-class screening prediction.

    Confidence is the max softmax probability. It is not disease severity
    and not the percentage of the eye that is affected.
    """
    if not app_state.get("model_loaded"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not loaded.",
        )

    contents = await file.read()
    filename = validate_upload(file.filename, file.content_type, contents)
    batch = preprocess_image_bytes(contents)

    try:
        prediction = run_prediction(batch)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction path failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction failed. Please try again.",
        ) from exc

    return {
        "success": True,
        "filename": filename,
        "predicted_class": prediction["predicted_class"],
        "predicted_label": prediction["predicted_label"],
        "confidence": prediction["confidence"],
        "confidence_percentage": prediction["confidence_percentage"],
        "class_probabilities": prediction["class_probabilities"],
        "severity_available": False,
        "message": PREDICTION_MESSAGE,
    }


@app.post("/explain", tags=["explainability"])
async def explain(file: UploadFile = File(..., description="Fundus / eye image (JPG, JPEG, or PNG)")) -> dict[str, Any]:
    """Return the same prediction as /predict plus Grad-CAM visualizations.

    Grad-CAM failures do not discard the prediction. Confidence is still not
    disease severity.
    """
    if not app_state.get("model_loaded"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not loaded.",
        )

    contents = await file.read()
    filename = validate_upload(file.filename, file.content_type, contents)
    batch = preprocess_image_bytes(contents)

    try:
        prediction = run_prediction(batch)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction path failed during explain")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction failed. Please try again.",
        ) from exc

    payload: dict[str, Any] = {
        "success": True,
        "filename": filename,
        "predicted_class": prediction["predicted_class"],
        "predicted_label": prediction["predicted_label"],
        "confidence": prediction["confidence"],
        "confidence_percentage": prediction["confidence_percentage"],
        "class_probabilities": prediction["class_probabilities"],
        "severity_available": False,
        "message": PREDICTION_MESSAGE,
        "gradcam_available": False,
        "gradcam": None,
        "gradcam_error": None,
    }

    layers = app_state.get("gradcam_layers")
    if not app_state.get("gradcam_ready") or layers is None:
        payload["gradcam_error"] = (
            "Grad-CAM is not available because the EfficientNetV2-B0 target layer "
            "could not be connected."
        )
    else:
        try:
            payload["gradcam"] = generate_gradcam_payload(
                layers,
                batch,
                prediction["predicted_class"],
                prediction["predicted_label"],
                prediction["confidence"],
                prediction["confidence_percentage"],
            )
            payload["gradcam_available"] = True
        except (GradCamError, Exception):
            logger.exception("Grad-CAM generation failed")
            payload["gradcam_available"] = False
            payload["gradcam"] = None
            payload["gradcam_error"] = (
                "The prediction succeeded, but Grad-CAM could not be generated for this image."
            )

    return payload


@app.post("/history", tags=["history"])
async def history_save(
    request: Request,
    file: UploadFile = File(..., description="Fundus / eye image (JPG, JPEG, or PNG)"),
    request_id: str = Form(...),
    predicted_class: int = Form(...),
    predicted_label: str = Form(...),
    confidence: float = Form(...),
    confidence_percentage: float = Form(...),
    class_0_probability: float = Form(...),
    class_1_probability: float = Form(...),
    class_2_probability: float = Form(...),
    class_3_probability: float = Form(...),
    class_4_probability: float = Form(...),
    heatmap_shape: str | None = Form(default=None),
    target_layer: str | None = Form(default=None),
    original: UploadFile | None = File(default=None),
    heatmap: UploadFile | None = File(default=None),
    overlay: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """Save a completed successful analysis. Duplicate request_id values update the same row."""
    request_id_value = str(request_id or "").strip()
    if not REQUEST_ID_RE.match(request_id_value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The analysis result could not be saved because the request identifier is invalid.",
        )

    contents = await file.read()
    filename = validate_upload(file.filename, file.content_type, contents)
    prediction = validate_history_prediction(
        predicted_class=predicted_class,
        predicted_label=predicted_label.strip() if isinstance(predicted_label, str) else "",
        confidence=confidence,
        confidence_percentage=confidence_percentage,
        class_probabilities={
            "Class 0": class_0_probability,
            "Class 1": class_1_probability,
            "Class 2": class_2_probability,
            "Class 3": class_3_probability,
            "Class 4": class_4_probability,
        },
    )

    original_bytes = await read_optional_image(original)
    heatmap_bytes = await read_optional_image(heatmap)
    overlay_bytes = await read_optional_image(overlay)
    analysis_id = persist_history(
        request_id=request_id_value,
        filename=filename,
        contents=contents,
        prediction=prediction,
        original_bytes=original_bytes,
        heatmap_bytes=heatmap_bytes,
        overlay_bytes=overlay_bytes,
        heatmap_shape=parse_heatmap_shape(heatmap_shape),
        target_layer=(target_layer or "").strip() or None,
    )
    if not analysis_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The analysis result could not be saved. Please try again.",
        )
    record = get_analysis(analysis_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The analysis result could not be saved. Please try again.",
        )
    return {
        "success": True,
        "id": analysis_id,
        "request_id": request_id_value,
        **serialize_history_summary(record),
        "uploaded_image": serialize_history_detail(request, record).get("uploaded_image"),
    }


@app.get("/history", tags=["history"])
def history_list(
    predicted_class: int | None = Query(default=None, ge=0, le=4),
    predicted_label: str | None = Query(default=None),
    q: str | None = Query(default=None, description="Filename search"),
    start_date: str | None = Query(default=None, description="Inclusive start date YYYY-MM-DD"),
    end_date: str | None = Query(default=None, description="Inclusive end date YYYY-MM-DD"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=8, ge=1, le=50),
) -> dict[str, Any]:
    class_id = resolve_class_filter(predicted_class, predicted_label)
    start_at, end_at = resolve_date_range(start_date, end_date)
    query = q.strip() if isinstance(q, str) and q.strip() else None
    try:
        listing = list_analyses(
            predicted_class=class_id,
            query=query,
            page=page,
            page_size=page_size,
            start_at=start_at,
            end_at=end_at,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to list analysis history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The screening service could not load analysis history. Please try again.",
        )
    items = [serialize_history_summary(item) for item in listing["items"]]
    return {
        "success": True,
        "items": items,
        "count": len(items),
        "total": listing["total"],
        "page": listing["page"],
        "page_size": listing["page_size"],
        "total_pages": listing["total_pages"],
    }


@app.get("/analytics/dashboard", tags=["analytics"])
def analytics_dashboard(
    predicted_class: int | None = Query(default=None, ge=0, le=4),
    predicted_label: str | None = Query(default=None),
    start_date: str | None = Query(default=None, description="Inclusive start date YYYY-MM-DD"),
    end_date: str | None = Query(default=None, description="Inclusive end date YYYY-MM-DD"),
) -> dict[str, Any]:
    class_id = resolve_class_filter(predicted_class, predicted_label)
    start_at, end_at = resolve_date_range(start_date, end_date)
    try:
        analytics = get_dashboard_analytics(
            predicted_class=class_id,
            start_at=start_at,
            end_at=end_at,
        )
    except Exception:
        logger.exception("Failed to load dashboard analytics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dashboard statistics could not be loaded. Please try again.",
        )
    return {
        "success": True,
        "severity_available": False,
        "message": PREDICTION_MESSAGE,
        **analytics,
    }


@app.get("/history/{analysis_id}", tags=["history"])
def history_detail(analysis_id: int, request: Request) -> dict[str, Any]:
    try:
        record = get_analysis(analysis_id)
    except Exception:
        logger.exception("Failed to load analysis detail")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="That analysis could not be loaded. Please try again.",
        )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="That analysis was not found.",
        )
    return serialize_history_detail(request, record)


@app.get("/history/{analysis_id}/image/{kind}", tags=["history"])
def history_image(analysis_id: int, kind: str):
    if kind not in {"uploaded", "original", "heatmap", "overlay"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown history image type.",
        )
    try:
        data = get_image_bytes(analysis_id, kind)
    except Exception:
        logger.exception("Failed to load history image")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="That image is not available for this analysis.",
        )
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="That image is not available for this analysis.",
        )
    payload, media = data
    return Response(content=payload, media_type=media)


