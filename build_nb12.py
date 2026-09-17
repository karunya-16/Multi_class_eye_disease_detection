"""Build notebooks/12_camera_classifier_external_validation.ipynb"""
import json
from pathlib import Path

NB = Path(r"D:\Practice Projects\Disease Detection\notebooks\12_camera_classifier_external_validation.ipynb")

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

cells = []

# ── Markdown title ─────────────────────────────────────────────────────────
cells.append(("markdown", "md_title", """\
# 12 — Camera Classifier External Validation

Externally validate `camera_binary_classifier_best.keras` on unseen camera eye images.

**This notebook does NOT:**
- retrain or modify any model
- modify the 4-class fundus model or its 84.94% accuracy
- modify the 5885-image fundus dataset or its CSVs
- modify Grad-CAM, FastAPI backend, or React frontend

**Class mapping (camera model):** `0 = Cataract` · `1 = Normal`

**Domain:** Normal vs Cataract only. DR and Glaucoma are NOT supported.
"""))

# ── Cell 1: Imports + config ───────────────────────────────────────────────
cells.append(("code", "cell_imports", """\
# CELL 1 - Imports, paths, config, protected-file fingerprints
from __future__ import annotations
import hashlib, json, warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
from PIL import Image, UnidentifiedImageError

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_fscore_support, f1_score,
)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── Project root ───────────────────────────────────────────────────────────
PROJECT_ROOT_CANDIDATES = [Path.cwd(), Path.cwd().parent,
                            Path(r"D:/Practice Projects/Disease Detection")]
PROJECT_ROOT = next(
    (p for p in PROJECT_ROOT_CANDIDATES
     if (p / "dataset").exists() and (p / "preprocessing").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate project root.")

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH   = PROJECT_ROOT / "model" / "camera_binary_classifier_best.keras"
REPORT_DIR   = PROJECT_ROOT / "reports" / "camera_classifier_external_validation"
GRADCAM_DIR  = REPORT_DIR / "gradcam"
PRED_CSV     = REPORT_DIR / "external_predictions.csv"
METRICS_JSON = REPORT_DIR / "external_validation_metrics.json"

for d in [REPORT_DIR, GRADCAM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── External validation folder (CONFIGURE HERE) ────────────────────────────
# Place unseen camera images under:
#   external_camera_validation/Cataract/  <- labelled Cataract images
#   external_camera_validation/Normal/    <- labelled Normal images
# Leave empty to skip evaluation gracefully.
EXT_DIR = PROJECT_ROOT / "external_camera_validation"

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
CLASS_MAP   = {0: "Cataract", 1: "Normal"}
CLASS_NAMES = ["Cataract", "Normal"]
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ── Protected-file fingerprints (BEFORE any writes) ────────────────────────
PROTECTED_RELATIVE = [
    "model/camera_binary_classifier_best.keras",
    "model/efficientnetv2_b0_4class_best.keras",
    "model/efficientnetv2_b0_4class_final.keras",
    "model/model_metadata_v2.json",
    "preprocessing/dataset_metadata_v2.csv",
    "preprocessing/splits_v2/train.csv",
    "preprocessing/splits_v2/validation.csv",
    "preprocessing/splits_v2/test.csv",
    "reports/model_v2/test_metrics.json",
    "backend/app/main.py",
    "backend/app/model_service.py",
    "backend/app/gradcam.py",
    "frontend/package.json",
]

def _fingerprint(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "size": None, "md5": None, "mtime_ns": None}
    data = path.read_bytes()
    return {"exists": True, "size": len(data),
            "md5": hashlib.md5(data).hexdigest(),
            "mtime_ns": path.stat().st_mtime_ns}

PROTECTED_BEFORE = {r: _fingerprint(PROJECT_ROOT / r) for r in PROTECTED_RELATIVE}

print("PROJECT_ROOT :", PROJECT_ROOT)
print("MODEL_PATH   :", MODEL_PATH, "| exists:", MODEL_PATH.exists())
print("EXT_DIR      :", EXT_DIR)
print("REPORT_DIR   :", REPORT_DIR)
print("TF version   :", tf.__version__)
print("GPU available:", bool(tf.config.list_physical_devices("GPU")))
print("Class map    :", CLASS_MAP)
print("Domain       : Normal vs Cataract ONLY (DR and Glaucoma NOT supported)")
print("Protected files fingerprinted:", len(PROTECTED_BEFORE))
"""))

# ── Cell 2: Load model ─────────────────────────────────────────────────────
cells.append(("code", "cell_load_model", """\
# CELL 2 - Load camera binary classifier (read-only, no retraining)
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Camera model not found: {MODEL_PATH}\\n"
        "Run notebook 11 first to train the camera classifier."
    )

camera_model = keras.models.load_model(str(MODEL_PATH))
print("Model loaded:", MODEL_PATH.name)
print("Input shape :", camera_model.input_shape)
print("Output shape:", camera_model.output_shape)
print("Trainable   : False (evaluation only — model weights are not modified)")

# Verify model fingerprint has not changed since cell 1
fp_now = _fingerprint(MODEL_PATH)
fp_before = PROTECTED_BEFORE["model/camera_binary_classifier_best.keras"]
if fp_before["exists"] and fp_now["md5"] != fp_before["md5"]:
    print("WARNING: camera model checksum changed since notebook start!")
else:
    print("Camera model checksum: OK (unchanged)")
"""))

# ── Cell 3: Discover external images ──────────────────────────────────────
cells.append(("code", "cell_discover", """\
# CELL 3 - Discover external images
def discover_images(ext_dir: Path) -> tuple[list[dict], int]:
    records = []
    invalid = 0
    for class_name in CLASS_NAMES:
        class_dir = ext_dir / class_name
        if not class_dir.exists():
            continue
        for fp in sorted(class_dir.iterdir()):
            if fp.suffix.lower() not in IMG_EXTS:
                continue
            try:
                with Image.open(fp) as img:
                    img.verify()
                records.append({"path": fp, "true_label": class_name,
                                 "true_id": CLASS_NAMES.index(class_name)})
            except Exception:
                invalid += 1
                print(f"  Invalid/unreadable: {fp.name}")
    return records, invalid

# Check if external folder exists and has images
ext_cataract_dir = EXT_DIR / "Cataract"
ext_normal_dir   = EXT_DIR / "Normal"

n_cataract = len([f for f in ext_cataract_dir.iterdir()
                  if f.suffix.lower() in IMG_EXTS]) if ext_cataract_dir.exists() else 0
n_normal   = len([f for f in ext_normal_dir.iterdir()
                  if f.suffix.lower() in IMG_EXTS]) if ext_normal_dir.exists() else 0
total_ext  = n_cataract + n_normal

print(f"External folder : {EXT_DIR}")
print(f"  Cataract images: {n_cataract}")
print(f"  Normal images  : {n_normal}")
print(f"  Total          : {total_ext}")

HAS_IMAGES = total_ext > 0

if not HAS_IMAGES:
    print()
    print("External validation images not provided yet - validation skipped.")
else:
    records, n_invalid = discover_images(EXT_DIR)
    print(f"  Invalid/unreadable: {n_invalid}")
    print(f"  Usable images     : {len(records)}")
"""))

# ── Cell 4: Run predictions ────────────────────────────────────────────────
cells.append(("code", "cell_predict", """\
# CELL 4 - Run predictions on external images
if not HAS_IMAGES:
    print("External validation images not provided yet - validation skipped.")
else:
    def predict_single(img_path: Path) -> tuple[float, float, int, str, float]:
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = preprocess_input(
            np.expand_dims(np.array(img, dtype="float32"), 0)
        )
        prob_normal   = float(camera_model.predict(arr, verbose=0)[0][0])
        prob_cataract = 1.0 - prob_normal
        class_id      = 1 if prob_normal >= 0.5 else 0
        disease_name  = CLASS_MAP[class_id]
        confidence    = max(prob_normal, prob_cataract)
        return prob_normal, prob_cataract, class_id, disease_name, confidence

    rows = []
    print(f"Running predictions on {len(records)} images...")
    for rec in records:
        try:
            p_norm, p_cat, pred_id, pred_name, conf = predict_single(rec["path"])
            correct = (pred_id == rec["true_id"])
            rows.append({
                "filename":        rec["path"].name,
                "true_label":      rec["true_label"],
                "true_id":         rec["true_id"],
                "predicted_label": pred_name,
                "predicted_id":    pred_id,
                "prob_normal":     round(p_norm, 4),
                "prob_cataract":   round(p_cat,  4),
                "confidence":      round(conf,   4),
                "correct":         correct,
            })
        except Exception as e:
            print(f"  Prediction error for {rec['path'].name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(str(PRED_CSV), index=False)
    print(f"Predictions saved to: {PRED_CSV}")
    print(df[["filename","true_label","predicted_label","confidence","correct"]].to_string(index=False))
"""))

nb["cells"] = []
for kind, cid, src in cells:
    cell = {"cell_type": kind, "id": cid, "metadata": {}, "source": src}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb["cells"].append(cell)

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written {len(nb['cells'])} cells (pass 1)")
