"""MySQL persistence for 4-class prediction history.

Credentials are loaded from backend/.env. Image bytes are stored on disk under
data/prediction_files/; MySQL keeps paths, not BLOBs.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

logger = logging.getLogger("eye_disease_api")

API_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = API_DIR / "data"
FILES_DIR = DATA_DIR / "prediction_files"
_lock = threading.Lock()

TABLE_NAME = "prediction_history"
CLASS_NAMES = {0: "Normal", 1: "Cataract", 2: "Diabetic Retinopathy", 3: "Glaucoma"}
PROB_COLUMNS = {
    0: "prob_normal",
    1: "prob_cataract",
    2: "prob_diabetic_retinopathy",
    3: "prob_glaucoma",
}

CREATE_DATABASE_SQL = (
    "CREATE DATABASE IF NOT EXISTS `{db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
)

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INT NOT NULL AUTO_INCREMENT,
    created_at DATETIME(6) NOT NULL,
    filename VARCHAR(255) NULL,
    disease VARCHAR(64) NOT NULL,
    class_id TINYINT NOT NULL,
    confidence DOUBLE NOT NULL,
    prob_normal DOUBLE NOT NULL,
    prob_cataract DOUBLE NOT NULL,
    prob_diabetic_retinopathy DOUBLE NOT NULL,
    prob_glaucoma DOUBLE NOT NULL,
    gradcam_available TINYINT NOT NULL DEFAULT 0,
    uploaded_image_path VARCHAR(512) NULL,
    gradcam_path VARCHAR(512) NULL,
    gradcam_overlay_path VARCHAR(512) NULL,
    gradcam_original_path VARCHAR(512) NULL,
    target_layer VARCHAR(128) NULL,
    heatmap_shape VARCHAR(64) NULL,
    PRIMARY KEY (id),
    INDEX idx_prediction_history_created_at (created_at),
    INDEX idx_prediction_history_class_id (class_id),
    INDEX idx_prediction_history_disease (disease)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

LIST_COLUMNS = """
    id, created_at, filename, disease, class_id, confidence,
    prob_normal, prob_cataract, prob_diabetic_retinopathy, prob_glaucoma,
    gradcam_available, uploaded_image_path, gradcam_path,
    gradcam_overlay_path, gradcam_original_path, target_layer, heatmap_shape
"""


def _load_env_file() -> None:
    env_path = API_DIR / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def mysql_settings() -> dict[str, Any]:
    _load_env_file()
    return {
        "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.environ.get("MYSQL_PORT", "3306")),
        "user": os.environ.get("MYSQL_USER", "root"),
        "password": os.environ.get("MYSQL_PASSWORD", ""),
        "database": os.environ.get("MYSQL_DATABASE", "eye_disease_detection"),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
        "autocommit": False,
    }


def _connect(*, with_database: bool = True) -> pymysql.connections.Connection:
    settings = mysql_settings()
    if not with_database:
        settings = {k: v for k, v in settings.items() if k != "database"}
        settings["cursorclass"] = DictCursor
        settings["autocommit"] = False
    return pymysql.connect(**settings)


def init_db() -> dict[str, Any]:
    settings = mysql_settings()
    db_name = str(settings["database"]).replace("`", "")
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = _connect(with_database=False)
        try:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_DATABASE_SQL.format(db=db_name))
            conn.commit()
        finally:
            conn.close()

        conn = _connect(with_database=True)
        try:
            with conn.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)
            conn.commit()
        finally:
            conn.close()
    logger.info(
        "MySQL ready at %s:%s/%s.%s",
        settings["host"],
        settings["port"],
        settings["database"],
        TABLE_NAME,
    )
    return ping()


def ping() -> dict[str, Any]:
    settings = mysql_settings()
    conn = _connect(with_database=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 AS ok, DATABASE() AS db_name")
            row = cursor.fetchone() or {}
            cursor.execute(
                """
                SELECT COUNT(*) AS table_count
                FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = %s
                """,
                (TABLE_NAME,),
            )
            table_row = cursor.fetchone() or {}
    finally:
        conn.close()
    return {
        "connected": True,
        "host": settings["host"],
        "port": settings["port"],
        "database": row.get("db_name") or settings["database"],
        "table": TABLE_NAME,
        "table_ready": int(table_row.get("table_count") or 0) > 0,
    }


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(FILES_DIR.resolve()).as_posix()


def _write_bytes(directory: Path, name: str, data: bytes) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / name
    dest.write_bytes(data)
    return _relative_path(dest)


def _decode_data_url(value: str | None) -> bytes | None:
    if not value or not isinstance(value, str) or "," not in value:
        return None
    if not value.startswith("data:"):
        return None
    try:
        return base64.b64decode(value.split(",", 1)[1])
    except (ValueError, TypeError):
        return None


def _iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def _probabilities_from_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        CLASS_NAMES[i]: float(row[PROB_COLUMNS[i]])
        for i in range(4)
    }


def _filters(
    *,
    disease: str | None = None,
    class_id: int | None = None,
    query: str | None = None,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if class_id is not None:
        clauses.append("class_id = %s")
        params.append(int(class_id))
    elif disease:
        clauses.append("disease = %s")
        params.append(str(disease))
    if query:
        clauses.append("filename LIKE %s")
        params.append(f"%{query}%")
    if start_at is not None:
        clauses.append("created_at >= %s")
        params.append(start_at)
    if end_at is not None:
        clauses.append("created_at <= %s")
        params.append(end_at)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _row_to_dict(row: dict[str, Any], *, include_images: bool = False) -> dict[str, Any]:
    probabilities = _probabilities_from_row(row)
    disease = row["disease"]
    class_id = int(row["class_id"])
    confidence = float(row["confidence"])
    heatmap_shape = None
    if row.get("heatmap_shape"):
        try:
            heatmap_shape = json.loads(row["heatmap_shape"])
        except json.JSONDecodeError:
            heatmap_shape = None
    payload: dict[str, Any] = {
        "id": int(row["id"]),
        "created_at": _iso_timestamp(row["created_at"]),
        "filename": row.get("filename"),
        "disease": disease,
        "class_id": class_id,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "predicted_label": disease,
        "predicted_class": class_id,
        "confidence_percentage": round(confidence * 100.0, 2),
        "class_probabilities": probabilities,
        "gradcam_available": bool(row.get("gradcam_available")),
        "gradcam_path": row.get("gradcam_path"),
        "target_layer": row.get("target_layer"),
        "heatmap_shape": heatmap_shape,
    }
    if include_images:
        payload["uploaded_image"] = _file_to_data_url(row.get("uploaded_image_path"))
        if payload["gradcam_available"]:
            payload["gradcam"] = {
                "original_image": _file_to_data_url(row.get("gradcam_original_path"))
                or payload["uploaded_image"],
                "heatmap_image": _file_to_data_url(row.get("gradcam_path")),
                "overlay_image": _file_to_data_url(row.get("gradcam_overlay_path")),
                "heatmap_shape": heatmap_shape,
                "target_layer": row.get("target_layer"),
            }
    return payload


def _safe_file_path(relative: str | None) -> Path | None:
    if not relative:
        return None
    candidate = (FILES_DIR / relative).resolve()
    try:
        candidate.relative_to(FILES_DIR.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _file_to_data_url(relative: str | None) -> str | None:
    path = _safe_file_path(relative)
    if path is None:
        return None
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def save_prediction(
    *,
    filename: str | None,
    image_bytes: bytes,
    result: dict[str, Any],
    gradcam: dict[str, Any] | None = None,
) -> int:
    item_id = uuid.uuid4().hex
    suffix = Path(filename or "upload.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        suffix = ".png"
    item_dir = FILES_DIR / item_id
    uploaded_path = _write_bytes(item_dir, f"uploaded{suffix}", image_bytes)

    heatmap_path = overlay_path = original_path = None
    target_layer = None
    heatmap_shape = None
    gradcam_available = 0
    if gradcam:
        heatmap_bytes = _decode_data_url(gradcam.get("heatmap_image"))
        overlay_bytes = _decode_data_url(gradcam.get("overlay_image"))
        original_bytes = _decode_data_url(gradcam.get("original_image"))
        if heatmap_bytes:
            heatmap_path = _write_bytes(item_dir, "heatmap.png", heatmap_bytes)
            gradcam_available = 1
        if overlay_bytes:
            overlay_path = _write_bytes(item_dir, "overlay.png", overlay_bytes)
        if original_bytes:
            original_path = _write_bytes(item_dir, "original.png", original_bytes)
        target_layer = gradcam.get("target_layer")
        if gradcam.get("heatmap_shape") is not None:
            heatmap_shape = json.dumps(gradcam.get("heatmap_shape"))

    probabilities = result["probabilities"]
    created_at = datetime.now(timezone.utc).replace(tzinfo=None)
    values = (
        created_at,
        filename,
        result["disease"],
        int(result["class_id"]),
        float(result["confidence"]),
        float(probabilities[CLASS_NAMES[0]]),
        float(probabilities[CLASS_NAMES[1]]),
        float(probabilities[CLASS_NAMES[2]]),
        float(probabilities[CLASS_NAMES[3]]),
        gradcam_available,
        uploaded_path,
        heatmap_path,
        overlay_path,
        original_path,
        target_layer,
        heatmap_shape,
    )
    sql = f"""
        INSERT INTO {TABLE_NAME} (
            created_at, filename, disease, class_id, confidence,
            prob_normal, prob_cataract, prob_diabetic_retinopathy, prob_glaucoma,
            gradcam_available, uploaded_image_path, gradcam_path,
            gradcam_overlay_path, gradcam_original_path, target_layer, heatmap_shape
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """
    with _lock:
        conn = _connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, values)
                record_id = int(cursor.lastrowid)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    logger.info("Saved prediction_history id=%s disease=%s", record_id, result["disease"])
    return record_id


def list_predictions(
    *,
    disease: str | None = None,
    class_id: int | None = None,
    query: str | None = None,
    page: int = 1,
    page_size: int = 8,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> dict[str, Any]:
    page = max(1, int(page or 1))
    page_size = min(50, max(1, int(page_size or 10)))
    where, params = _filters(
        disease=disease,
        class_id=class_id,
        query=query,
        start_at=start_at,
        end_at=end_at,
    )
    with _lock:
        conn = _connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) AS total FROM {TABLE_NAME}{where}", params)
                total = int((cursor.fetchone() or {}).get("total") or 0)
                offset = (page - 1) * page_size
                cursor.execute(
                    f"SELECT {LIST_COLUMNS} FROM {TABLE_NAME}{where} "
                    "ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
                    [*params, page_size, offset],
                )
                rows = cursor.fetchall() or []
        finally:
            conn.close()
    items = [_row_to_dict(row) for row in rows]
    total_pages = max(1, (total + page_size - 1) // page_size) if total else 1
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


def get_prediction(record_id: int) -> dict[str, Any] | None:
    with _lock:
        conn = _connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT {LIST_COLUMNS} FROM {TABLE_NAME} WHERE id = %s",
                    (int(record_id),),
                )
                row = cursor.fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    return _row_to_dict(row, include_images=True)


def dashboard_stats(
    *,
    disease: str | None = None,
    class_id: int | None = None,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    recent_start = now - timedelta(days=7)
    if start_at is not None and start_at > recent_start:
        recent_start = start_at
    where, params = _filters(disease=disease, class_id=class_id, start_at=start_at, end_at=end_at)
    recent_where, recent_params = _filters(
        disease=disease,
        class_id=class_id,
        start_at=recent_start,
        end_at=end_at,
    )
    with _lock:
        conn = _connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        COUNT(*) AS total_predictions,
                        AVG(confidence) AS average_confidence,
                        MAX(confidence) AS highest_confidence,
                        MIN(confidence) AS lowest_confidence,
                        SUM(gradcam_available) AS gradcam_count
                    FROM {TABLE_NAME}{where}
                    """,
                    params,
                )
                summary = cursor.fetchone() or {}
                cursor.execute(
                    f"""
                    SELECT class_id, disease, COUNT(*) AS class_count
                    FROM {TABLE_NAME}{where}
                    GROUP BY class_id, disease
                    """,
                    params,
                )
                class_rows = cursor.fetchall() or []
                cursor.execute(
                    f"""
                    SELECT
                        SUM(CASE WHEN confidence < 0.50 THEN 1 ELSE 0 END) AS bucket_0_50,
                        SUM(CASE WHEN confidence >= 0.50 AND confidence < 0.70 THEN 1 ELSE 0 END) AS bucket_50_70,
                        SUM(CASE WHEN confidence >= 0.70 AND confidence < 0.85 THEN 1 ELSE 0 END) AS bucket_70_85,
                        SUM(CASE WHEN confidence >= 0.85 THEN 1 ELSE 0 END) AS bucket_85_100
                    FROM {TABLE_NAME}{where}
                    """,
                    params,
                )
                buckets = cursor.fetchone() or {}
                cursor.execute(
                    f"SELECT COUNT(*) AS recent_count FROM {TABLE_NAME}{recent_where}",
                    recent_params,
                )
                recent = cursor.fetchone() or {}
                cursor.execute(
                    f"""
                    SELECT DATE(created_at) AS bucket, COUNT(*) AS analysis_count
                    FROM {TABLE_NAME}{where}
                    GROUP BY DATE(created_at)
                    ORDER BY bucket
                    """,
                    params,
                )
                activity_rows = cursor.fetchall() or []
        finally:
            conn.close()

    counts = {int(row["class_id"]): int(row["class_count"] or 0) for row in class_rows}
    class_distribution = [
        {
            "class_id": class_id,
            "disease": CLASS_NAMES[class_id],
            "predicted_class": class_id,
            "predicted_label": CLASS_NAMES[class_id],
            "count": counts.get(class_id, 0),
        }
        for class_id in range(4)
    ]
    total = int(summary.get("total_predictions") or 0)
    avg = float(summary["average_confidence"]) if total and summary.get("average_confidence") is not None else None
    high = float(summary["highest_confidence"]) if total and summary.get("highest_confidence") is not None else None
    low = float(summary["lowest_confidence"]) if total and summary.get("lowest_confidence") is not None else None
    return {
        "total_predictions": total,
        "total_analyses": total,
        "recent_count": int(recent.get("recent_count") or 0),
        "recent_analysis_count": int(recent.get("recent_count") or 0),
        "recent_window_days": 7,
        "average_confidence": None if avg is None else round(avg, 4),
        "average_confidence_percentage": None if avg is None else round(avg * 100.0, 2),
        "highest_confidence": None if high is None else round(high, 4),
        "highest_confidence_percentage": None if high is None else round(high * 100.0, 2),
        "lowest_confidence": None if low is None else round(low, 4),
        "lowest_confidence_percentage": None if low is None else round(low * 100.0, 2),
        "gradcam_count": int(summary.get("gradcam_count") or 0),
        "class_distribution": class_distribution,
        "confidence_distribution": [
            {"bucket": "0-50%", "count": int(buckets.get("bucket_0_50") or 0)},
            {"bucket": "50-70%", "count": int(buckets.get("bucket_50_70") or 0)},
            {"bucket": "70-85%", "count": int(buckets.get("bucket_70_85") or 0)},
            {"bucket": "85-100%", "count": int(buckets.get("bucket_85_100") or 0)},
        ],
        "activity": [
            {
                "bucket": str(row.get("bucket")),
                "count": int(row.get("analysis_count") or 0),
            }
            for row in activity_rows
            if row.get("bucket") is not None
        ],
        "activity_granularity": "day",
    }
