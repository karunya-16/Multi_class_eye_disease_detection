"""MySQL storage for completed analysis results.

Images are written to `data/history_files/` and only file references are stored
in MySQL. Credentials come from backend environment variables, never the frontend.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

logger = logging.getLogger("eye_disease_api")

API_DIR = Path(__file__).resolve().parent
DATA_DIR = API_DIR / "data"
FILES_DIR = DATA_DIR / "history_files"
_lock = threading.Lock()

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
TABLE_NAME = "analysis_results"

CREATE_DATABASE_SQL = (
    "CREATE DATABASE IF NOT EXISTS `{db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
)

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INT NOT NULL AUTO_INCREMENT,
    request_id VARCHAR(64) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    filename VARCHAR(255) NULL,
    image_path VARCHAR(512) NULL,
    predicted_class INT NOT NULL,
    predicted_label VARCHAR(32) NOT NULL,
    confidence DOUBLE NOT NULL,
    confidence_percentage DOUBLE NOT NULL,
    class_0_probability DOUBLE NOT NULL,
    class_1_probability DOUBLE NOT NULL,
    class_2_probability DOUBLE NOT NULL,
    class_3_probability DOUBLE NOT NULL,
    class_4_probability DOUBLE NOT NULL,
    gradcam_available TINYINT NOT NULL DEFAULT 0,
    gradcam_original_path VARCHAR(512) NULL,
    gradcam_heatmap_path VARCHAR(512) NULL,
    gradcam_overlay_path VARCHAR(512) NULL,
    heatmap_shape VARCHAR(64) NULL,
    target_layer VARCHAR(128) NULL,
    severity_available TINYINT NOT NULL DEFAULT 0,
    PRIMARY KEY (id),
    UNIQUE KEY uq_analysis_results_request_id (request_id),
    INDEX idx_analysis_results_created_at (created_at),
    INDEX idx_analysis_results_predicted_class (predicted_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

LIST_COLUMNS = """
    id, request_id, created_at, filename, image_path, predicted_class,
    predicted_label, confidence, confidence_percentage,
    class_0_probability, class_1_probability, class_2_probability,
    class_3_probability, class_4_probability, gradcam_available,
    gradcam_original_path, gradcam_heatmap_path, gradcam_overlay_path,
    heatmap_shape, target_layer, severity_available
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


def _table_exists(conn: pymysql.connections.Connection, name: str) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*) AS table_count
            FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = %s
            """,
            (name,),
        )
        row = cursor.fetchone()
    return bool(row and row["table_count"])


def init_history_store() -> None:
    settings = mysql_settings()
    db_name = settings["database"].replace("`", "")
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
            if not _table_exists(conn, TABLE_NAME):
                with conn.cursor() as cursor:
                    cursor.execute(CREATE_TABLE_SQL)
                conn.commit()
                logger.info("Created MySQL table %s", TABLE_NAME)
            else:
                logger.info("MySQL table %s already exists", TABLE_NAME)
            _ensure_indexes(conn)
            conn.commit()
        finally:
            conn.close()
    logger.info(
        "MySQL analysis history ready at %s:%s/%s.%s",
        settings["host"],
        settings["port"],
        settings["database"],
        TABLE_NAME,
    )


def _ensure_indexes(conn: pymysql.connections.Connection) -> None:
    wanted = {
        "idx_analysis_results_created_class": (
            f"CREATE INDEX idx_analysis_results_created_class "
            f"ON {TABLE_NAME} (created_at, predicted_class)"
        ),
        "idx_analysis_results_class_created": (
            f"CREATE INDEX idx_analysis_results_class_created "
            f"ON {TABLE_NAME} (predicted_class, created_at)"
        ),
    }
    with conn.cursor() as cursor:
        for name, ddl in wanted.items():
            cursor.execute(
                """
                SELECT COUNT(*) AS index_count
                FROM information_schema.statistics
                WHERE table_schema = DATABASE()
                  AND table_name = %s
                  AND index_name = %s
                """,
                (TABLE_NAME, name),
            )
            row = cursor.fetchone()
            if row and int(row["index_count"] or 0) > 0:
                continue
            cursor.execute(ddl)
            logger.info("Created MySQL index %s", name)


def _analysis_filters(
    predicted_class: int | None = None,
    predicted_label: str | None = None,
    query: str | None = None,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if predicted_class is not None:
        clauses.append("predicted_class = %s")
        params.append(int(predicted_class))
    elif predicted_label:
        clauses.append("predicted_label = %s")
        params.append(str(predicted_label))
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


def normalize_request_id(raw: str | None) -> str:
    value = str(raw or "").strip()
    if REQUEST_ID_RE.match(value):
        return value
    return uuid.uuid4().hex


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(FILES_DIR.resolve()).as_posix()


def _write_bytes(directory: Path, name: str, data: bytes) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / name
    dest.write_bytes(data)
    return _relative_path(dest)


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


def _iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


def save_analysis(
    *,
    request_id: str,
    filename: str,
    uploaded_bytes: bytes,
    predicted_class: int,
    predicted_label: str,
    confidence: float,
    confidence_percentage: float,
    class_probabilities: dict[str, Any],
    original_bytes: bytes | None = None,
    heatmap_bytes: bytes | None = None,
    overlay_bytes: bytes | None = None,
    heatmap_shape: Any = None,
    target_layer: str | None = None,
) -> int | None:
    """Insert or update one successful analysis. Never called for failed predictions."""
    try:
        request_id = normalize_request_id(request_id)
        suffix = Path(filename or "upload.png").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            suffix = ".png"
        item_dir = FILES_DIR / request_id
        image_path = _write_bytes(item_dir, f"uploaded{suffix}", uploaded_bytes)

        gradcam_original = gradcam_heatmap = gradcam_overlay = None
        gradcam_available = 0
        shape_text = json.dumps(heatmap_shape) if heatmap_shape else None
        if heatmap_bytes and overlay_bytes:
            if original_bytes:
                gradcam_original = _write_bytes(item_dir, "original.png", original_bytes)
            gradcam_heatmap = _write_bytes(item_dir, "heatmap.png", heatmap_bytes)
            gradcam_overlay = _write_bytes(item_dir, "overlay.png", overlay_bytes)
            gradcam_available = 1

        created_at = datetime.now(timezone.utc).replace(tzinfo=None)
        values = (
            request_id,
            created_at,
            filename,
            image_path,
            int(predicted_class),
            predicted_label,
            float(confidence),
            float(confidence_percentage),
            float(class_probabilities["Class 0"]),
            float(class_probabilities["Class 1"]),
            float(class_probabilities["Class 2"]),
            float(class_probabilities["Class 3"]),
            float(class_probabilities["Class 4"]),
            gradcam_available,
            gradcam_original,
            gradcam_heatmap,
            gradcam_overlay,
            shape_text,
            target_layer,
        )
        sql = f"""
            INSERT INTO {TABLE_NAME} (
                request_id, created_at, filename, image_path, predicted_class,
                predicted_label, confidence, confidence_percentage,
                class_0_probability, class_1_probability, class_2_probability,
                class_3_probability, class_4_probability, gradcam_available,
                gradcam_original_path, gradcam_heatmap_path, gradcam_overlay_path,
                heatmap_shape, target_layer, severity_available
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0
            )
            ON DUPLICATE KEY UPDATE
                id = LAST_INSERT_ID(id),
                filename = VALUES(filename),
                image_path = VALUES(image_path),
                predicted_class = VALUES(predicted_class),
                predicted_label = VALUES(predicted_label),
                confidence = VALUES(confidence),
                confidence_percentage = VALUES(confidence_percentage),
                class_0_probability = VALUES(class_0_probability),
                class_1_probability = VALUES(class_1_probability),
                class_2_probability = VALUES(class_2_probability),
                class_3_probability = VALUES(class_3_probability),
                class_4_probability = VALUES(class_4_probability),
                heatmap_shape = IF(VALUES(gradcam_available) = 1, VALUES(heatmap_shape), heatmap_shape),
                target_layer = IF(VALUES(gradcam_available) = 1, VALUES(target_layer), target_layer),
                gradcam_available = IF(VALUES(gradcam_available) = 1, 1, gradcam_available),
                gradcam_original_path = IF(VALUES(gradcam_available) = 1, VALUES(gradcam_original_path), gradcam_original_path),
                gradcam_heatmap_path = IF(VALUES(gradcam_available) = 1, VALUES(gradcam_heatmap_path), gradcam_heatmap_path),
                gradcam_overlay_path = IF(VALUES(gradcam_available) = 1, VALUES(gradcam_overlay_path), gradcam_overlay_path)
        """
        with _lock:
            conn = _connect()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql, values)
                    analysis_id = int(cursor.lastrowid or 0)
                    if analysis_id <= 0:
                        cursor.execute(
                            f"SELECT id FROM {TABLE_NAME} WHERE request_id = %s",
                            (request_id,),
                        )
                        found = cursor.fetchone()
                        analysis_id = int(found["id"]) if found else 0
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        logger.info("Saved analysis_results id=%s request_id=%s", analysis_id, request_id)
        return analysis_id
    except Exception:
        logger.exception("Failed to save analysis history to MySQL")
        return None


def _row_to_dict(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        probabilities = {
            "Class 0": float(row["class_0_probability"]),
            "Class 1": float(row["class_1_probability"]),
            "Class 2": float(row["class_2_probability"]),
            "Class 3": float(row["class_3_probability"]),
            "Class 4": float(row["class_4_probability"]),
        }
        heatmap_shape = None
        if row.get("heatmap_shape"):
            heatmap_shape = json.loads(row["heatmap_shape"])
        return {
            "id": int(row["id"]),
            "request_id": row["request_id"],
            "created_at": _iso_timestamp(row["created_at"]),
            "filename": row["filename"],
            "image_path": row["image_path"],
            "predicted_class": int(row["predicted_class"]),
            "predicted_label": row["predicted_label"],
            "confidence": float(row["confidence"]),
            "confidence_percentage": float(row["confidence_percentage"]),
            "class_probabilities": probabilities,
            "severity_available": False,
            "gradcam_available": bool(row["gradcam_available"]),
            "gradcam_original_path": row.get("gradcam_original_path"),
            "gradcam_heatmap_path": row.get("gradcam_heatmap_path"),
            "gradcam_overlay_path": row.get("gradcam_overlay_path"),
            "heatmap_shape": heatmap_shape,
            "target_layer": row.get("target_layer"),
            "has_uploaded": bool(row.get("image_path")),
            "has_heatmap": bool(row.get("gradcam_heatmap_path")),
            "has_overlay": bool(row.get("gradcam_overlay_path")),
        }
    except (TypeError, ValueError, json.JSONDecodeError, KeyError):
        logger.warning("Skipping invalid analysis_results row id=%s", row.get("id") if row else None)
        return None


def list_analyses(
    predicted_class: int | None = None,
    predicted_label: str | None = None,
    query: str | None = None,
    page: int = 1,
    page_size: int = 8,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> dict[str, Any]:
    page = max(1, int(page or 1))
    page_size = min(50, max(1, int(page_size or 10)))
    where, params = _analysis_filters(
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        query=query,
        start_at=start_at,
        end_at=end_at,
    )
    count_sql = f"SELECT COUNT(*) AS total FROM {TABLE_NAME}{where}"
    list_sql = (
        f"SELECT {LIST_COLUMNS} FROM {TABLE_NAME}{where} "
        "ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s"
    )
    with _lock:
        conn = _connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(count_sql, params)
                total = int((cursor.fetchone() or {}).get("total") or 0)
                offset = (page - 1) * page_size
                cursor.execute(list_sql, [*params, page_size, offset])
                rows = cursor.fetchall()
        finally:
            conn.close()
    items = [parsed for row in rows if (parsed := _row_to_dict(row))]
    total_pages = max(1, (total + page_size - 1) // page_size) if total else 1
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


def get_analysis(analysis_id: int) -> dict[str, Any] | None:
    with _lock:
        conn = _connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"SELECT {LIST_COLUMNS} FROM {TABLE_NAME} WHERE id = %s",
                    (int(analysis_id),),
                )
                row = cursor.fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    return _row_to_dict(row)


def get_image_bytes(analysis_id: int, kind: str) -> tuple[bytes, str] | None:
    record = get_analysis(analysis_id)
    if record is None:
        return None
    path_key = {
        "uploaded": "image_path",
        "original": "gradcam_original_path",
        "heatmap": "gradcam_heatmap_path",
        "overlay": "gradcam_overlay_path",
    }.get(kind)
    if not path_key:
        return None
    relative = record.get(path_key)
    if kind == "original" and not relative:
        relative = record.get("image_path")
    path = _safe_file_path(relative)
    if path is None:
        return None
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return path.read_bytes(), mime


CLASS_LABELS = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4",
}
CONFIDENCE_BUCKETS = (
    ("0-50%", 0.0, 0.50),
    ("50-70%", 0.50, 0.70),
    ("70-85%", 0.70, 0.85),
    ("85-100%", 0.85, 1.01),
)
RECENT_WINDOW_DAYS = 7


def _round_or_none(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return round(number, digits)


def _bucket_label(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:00")
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    if len(text) >= 19 and text[10] == " ":
        return text[:13] + ":00"
    if " " in text:
        return text.split(" ")[0]
    return text[:10]


def get_dashboard_analytics(
    predicted_class: int | None = None,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> dict[str, Any]:
    where, params = _analysis_filters(
        predicted_class=predicted_class,
        start_at=start_at,
        end_at=end_at,
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    recent_start = now - timedelta(days=RECENT_WINDOW_DAYS)
    if start_at is not None and start_at > recent_start:
        recent_start = start_at
    recent_where, recent_params = _analysis_filters(
        predicted_class=predicted_class,
        start_at=recent_start,
        end_at=end_at,
    )

    summary_sql = f"""
        SELECT
            COUNT(*) AS total_analyses,
            AVG(confidence) AS average_confidence,
            MAX(confidence) AS highest_confidence,
            MIN(confidence) AS lowest_confidence,
            AVG(confidence_percentage) AS average_confidence_percentage,
            MAX(confidence_percentage) AS highest_confidence_percentage,
            MIN(confidence_percentage) AS lowest_confidence_percentage
        FROM {TABLE_NAME}{where}
    """
    class_sql = f"""
        SELECT predicted_class, COUNT(*) AS class_count
        FROM {TABLE_NAME}{where}
        GROUP BY predicted_class
    """
    confidence_sql = f"""
        SELECT
            SUM(CASE WHEN confidence < 0.50 THEN 1 ELSE 0 END) AS bucket_0_50,
            SUM(CASE WHEN confidence >= 0.50 AND confidence < 0.70 THEN 1 ELSE 0 END) AS bucket_50_70,
            SUM(CASE WHEN confidence >= 0.70 AND confidence < 0.85 THEN 1 ELSE 0 END) AS bucket_70_85,
            SUM(CASE WHEN confidence >= 0.85 THEN 1 ELSE 0 END) AS bucket_85_100
        FROM {TABLE_NAME}{where}
    """
    recent_sql = f"SELECT COUNT(*) AS recent_count FROM {TABLE_NAME}{recent_where}"

    span_hours = None
    if start_at is not None and end_at is not None:
        span_hours = max(0.0, (end_at - start_at).total_seconds() / 3600.0)
    if span_hours is not None and span_hours <= 48:
        granularity = "hour"
        activity_select = "DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:00:00')"
    else:
        granularity = "day"
        activity_select = "DATE(created_at)"
    activity_sql = f"""
        SELECT {activity_select} AS bucket, COUNT(*) AS analysis_count
        FROM {TABLE_NAME}{where}
        GROUP BY {activity_select}
        ORDER BY bucket
    """

    with _lock:
        conn = _connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(summary_sql, params)
                summary = cursor.fetchone() or {}
                cursor.execute(class_sql, params)
                class_rows = cursor.fetchall() or []
                cursor.execute(confidence_sql, params)
                confidence_row = cursor.fetchone() or {}
                cursor.execute(recent_sql, recent_params)
                recent_row = cursor.fetchone() or {}
                cursor.execute(activity_sql, params)
                activity_rows = cursor.fetchall() or []
        finally:
            conn.close()

    counts_by_class = {
        int(row["predicted_class"]): int(row["class_count"] or 0)
        for row in class_rows
        if row.get("predicted_class") is not None
    }
    class_distribution = [
        {
            "predicted_class": class_id,
            "predicted_label": CLASS_LABELS[class_id],
            "count": counts_by_class.get(class_id, 0),
        }
        for class_id in range(5)
    ]
    bucket_values = [
        int(confidence_row.get("bucket_0_50") or 0),
        int(confidence_row.get("bucket_50_70") or 0),
        int(confidence_row.get("bucket_70_85") or 0),
        int(confidence_row.get("bucket_85_100") or 0),
    ]
    confidence_distribution = [
        {
            "bucket": label,
            "min_confidence": low,
            "max_confidence": min(1.0, high),
            "count": count,
        }
        for (label, low, high), count in zip(CONFIDENCE_BUCKETS, bucket_values)
    ]
    activity = [
        {
            "bucket": _bucket_label(row.get("bucket")),
            "count": int(row.get("analysis_count") or 0),
        }
        for row in activity_rows
        if row.get("bucket") is not None
    ]
    total = int(summary.get("total_analyses") or 0)
    return {
        "total_analyses": total,
        "recent_analysis_count": int(recent_row.get("recent_count") or 0),
        "recent_window_days": RECENT_WINDOW_DAYS,
        "average_confidence": _round_or_none(summary.get("average_confidence"), 4) if total else None,
        "average_confidence_percentage": _round_or_none(
            summary.get("average_confidence_percentage"), 2
        )
        if total
        else None,
        "highest_confidence": _round_or_none(summary.get("highest_confidence"), 4) if total else None,
        "highest_confidence_percentage": _round_or_none(
            summary.get("highest_confidence_percentage"), 2
        )
        if total
        else None,
        "lowest_confidence": _round_or_none(summary.get("lowest_confidence"), 4) if total else None,
        "lowest_confidence_percentage": _round_or_none(
            summary.get("lowest_confidence_percentage"), 2
        )
        if total
        else None,
        "class_distribution": class_distribution,
        "confidence_distribution": confidence_distribution,
        "activity": activity,
        "activity_granularity": granularity,
    }

