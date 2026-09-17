"""Patch notebook 12 cells 3, 4, 8, 9, 10 for recursive Cataract discovery."""
import json

NB_PATH = r"d:\Practice Projects\Disease Detection\notebooks\12_camera_classifier_external_validation.ipynb"

# ── CELL 3 ─────────────────────────────────────────────────────────────────
CELL3 = """\
# CELL 3 - Discover external images (recursive: Cataract/immature, Cataract/mature, Normal)
EXPECTED_CATARACT = 205
EXPECTED_NORMAL   = 94
EXPECTED_TOTAL    = 299

def discover_images(ext_dir):
    \"\"\"Recursively discover images; Cataract sub-folders map to true_id=0.\"\"\"
    records = []
    invalid = 0

    # --- Cataract (recurse into immature/ and mature/) ---
    cat_dir = ext_dir / "Cataract"
    if cat_dir.exists():
        for fp in sorted(cat_dir.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                rel = fp.relative_to(ext_dir)   # e.g. Cataract/immature/img.jpg
                try:
                    with Image.open(fp) as img:
                        img.verify()
                    records.append({
                        "path":          fp,
                        "relative_path": str(rel).replace("\\\\", "/"),
                        "true_label":    "Cataract",
                        "true_id":       0,
                    })
                except Exception:
                    invalid += 1
                    print(f"  Invalid/unreadable: {rel}")

    # --- Normal (flat) ---
    norm_dir = ext_dir / "Normal"
    if norm_dir.exists():
        for fp in sorted(norm_dir.iterdir()):
            if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                rel = fp.relative_to(ext_dir)   # e.g. Normal/img.jpg
                try:
                    with Image.open(fp) as img:
                        img.verify()
                    records.append({
                        "path":          fp,
                        "relative_path": str(rel).replace("\\\\", "/"),
                        "true_label":    "Normal",
                        "true_id":       1,
                    })
                except Exception:
                    invalid += 1
                    print(f"  Invalid/unreadable: {rel}")

    return records, invalid


print(f"External folder : {EXT_DIR}")

records, n_invalid = discover_images(EXT_DIR)

n_cataract = sum(1 for r in records if r["true_id"] == 0)
n_normal   = sum(1 for r in records if r["true_id"] == 1)
total_ext  = len(records)

print(f"Expected external images: {EXPECTED_TOTAL}")
print(f"Discovered usable images: {total_ext}")
print(f"  Cataract: {n_cataract}")
print(f"  Normal  : {n_normal}")
print(f"  Invalid/unreadable: {n_invalid}")

if n_cataract != EXPECTED_CATARACT or n_normal != EXPECTED_NORMAL:
    print()
    print(f"WARNING: Expected {EXPECTED_CATARACT} Cataract and {EXPECTED_NORMAL} Normal images.")
    print(f"         Found {n_cataract} Cataract and {n_normal} Normal.")
    print("         Validation may be incomplete.")

HAS_IMAGES = total_ext > 0
"""

# ── CELL 4 ─────────────────────────────────────────────────────────────────
CELL4 = """\
# CELL 4 - Run predictions on external images
if not HAS_IMAGES:
    print("External validation images not provided yet - validation skipped.")
else:
    def predict_single(img_path):
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
                "relative_path":   rec["relative_path"],
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
            print(f"  Prediction error for {rec['relative_path']}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(str(PRED_CSV), index=False)
    print(f"Predictions saved to: {PRED_CSV}")
    print(df[["relative_path","true_label","predicted_label","confidence","correct"]].to_string(index=False))
"""

# ── CELL 8 ─────────────────────────────────────────────────────────────────
CELL8 = """\
# CELL 8 - Representative correct and incorrect predictions
import math as _math

def show_ext_samples(subset_df, title, save_name, n=8):
    subset_df = subset_df.head(n)
    if len(subset_df) == 0:
        print(f"No samples for: {title}"); return
    cols = min(4, len(subset_df))
    rows = _math.ceil(len(subset_df) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for i, (_, row) in enumerate(subset_df.iterrows()):
        # Use relative_path to reconstruct full path (handles nested Cataract sub-folders)
        img_path = EXT_DIR / row["relative_path"]
        try:
            img = Image.open(img_path).resize((224, 224))
            axes[i].imshow(img)
        except Exception:
            axes[i].text(0.5, 0.5, "load error", ha="center", va="center")
        color = "green" if row["correct"] else "red"
        axes[i].set_title(
            f"T:{row['true_label']}\\nP:{row['predicted_label']} ({row['confidence']:.2f})",
            fontsize=8, color=color)
        axes[i].axis("off")
    for j in range(len(subset_df), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(str(REPORT_DIR / save_name), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_name}")

if not HAS_IMAGES:
    print("External validation images not provided yet - validation skipped.")
else:
    show_ext_samples(df[df["correct"]],  "Correct Predictions (External)",   "ext_correct_predictions.png")
    show_ext_samples(df[~df["correct"]], "Incorrect Predictions (External)", "ext_incorrect_predictions.png")
"""

# ── CELL 9 ─────────────────────────────────────────────────────────────────
CELL9 = """\
# CELL 9 - Grad-CAM on external images (camera model only)
# Does NOT modify backend/app/gradcam.py
import cv2

def get_gradcam_heatmap(mdl, img_array, conv_layer_name):
    grad_model = Model(
        inputs=mdl.inputs,
        outputs=[mdl.get_layer(conv_layer_name).output, mdl.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = np.array(Image.open(img_path).resize((224, 224)).convert("RGB"))
    h   = cv2.resize(heatmap, (224, 224))
    hc  = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return img, (hc * alpha + img * (1 - alpha)).astype(np.uint8)

def find_last_conv(mdl):
    for layer in reversed(mdl.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    return None

def run_gradcam_ext(subset_df, label, n=4):
    last_conv = find_last_conv(camera_model)
    if last_conv is None:
        print("Could not find conv layer for Grad-CAM"); return
    subset_df = subset_df.head(n)
    if len(subset_df) == 0:
        print(f"No samples for Grad-CAM: {label}"); return
    fig, axes = plt.subplots(len(subset_df), 2, figsize=(8, len(subset_df) * 3))
    if len(subset_df) == 1: axes = [axes]
    for row_i, (_, row) in enumerate(subset_df.iterrows()):
        # Use relative_path to reconstruct full path (handles nested Cataract sub-folders)
        img_path = EXT_DIR / row["relative_path"]
        arr = preprocess_input(
            np.expand_dims(
                np.array(Image.open(img_path).resize((224,224)).convert("RGB")), 0
            ).astype("float32")
        )
        try:
            hm = get_gradcam_heatmap(camera_model, arr, last_conv)
            orig, ov = overlay_gradcam(img_path, hm)
            axes[row_i][0].imshow(orig);  axes[row_i][0].set_title("Original"); axes[row_i][0].axis("off")
            axes[row_i][1].imshow(ov);    axes[row_i][1].set_title("Grad-CAM"); axes[row_i][1].axis("off")
        except Exception as e:
            axes[row_i][0].set_title(str(e)); axes[row_i][0].axis("off"); axes[row_i][1].axis("off")
    fig.suptitle(f"Grad-CAM: {label}", fontsize=11)
    plt.tight_layout()
    sname = f"gradcam_{label.replace(' ','_')}.png"
    plt.savefig(str(GRADCAM_DIR / sname), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: gradcam/{sname}")

if not HAS_IMAGES:
    print("External validation images not provided yet - validation skipped.")
else:
    run_gradcam_ext(df[(df["true_id"]==0) & df["correct"]],  "correct_cataract", n=4)
    run_gradcam_ext(df[(df["true_id"]==1) & df["correct"]],  "correct_normal",   n=4)
    run_gradcam_ext(df[~df["correct"]],                      "incorrect",        n=4)
"""

# ── CELL 10 ────────────────────────────────────────────────────────────────
CELL10 = """\
# CELL 10 - Save metrics JSON
if not HAS_IMAGES:
    metrics = {
        "status": "SKIPPED",
        "reason": "External validation images not provided yet.",
        "ext_dir": str(EXT_DIR),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
else:
    metrics = {
        "status": "COMPLETED",
        "ext_dir": str(EXT_DIR),
        "domain": "Normal vs Cataract only. DR and Glaucoma NOT supported.",
        "class_mapping": {"0": "Cataract", "1": "Normal"},
        "images_evaluated":      len(df),
        "cataract_count":        int((y_true == 0).sum()),
        "normal_count":          int((y_true == 1).sum()),
        "invalid_images":        n_invalid,
        "incorrect_predictions": int((~df["correct"]).sum()),
        "accuracy":              round(float(acc),      4),
        "balanced_accuracy":     round(float(bal_acc),  4),
        "macro_f1":              round(float(macro_f1), 4),
        "roc_auc":               round(float(roc_auc),  4) if roc_auc is not None else None,
        "per_class": {
            "Cataract": {
                "precision": round(float(prec[0]), 4),
                "recall":    round(float(rec[0]),  4),
                "f1":        round(float(f1s[0]),  4),
                "support":   int(sup[0]),
            },
            "Normal": {
                "precision": round(float(prec[1]), 4),
                "recall":    round(float(rec[1]),  4),
                "f1":        round(float(f1s[1]),  4),
                "support":   int(sup[1]),
            },
        },
        "confusion_matrix": cm_arr.tolist(),
        "confidence_stats": {
            "all_mean":       round(float(conf_all.mean()),          4),
            "all_median":     round(float(np.median(conf_all)),      4),
            "all_min":        round(float(conf_all.min()),           4),
            "all_max":        round(float(conf_all.max()),           4),
            "correct_mean":   round(float(conf_correct.mean()),      4) if len(conf_correct) else None,
            "incorrect_mean": round(float(conf_wrong.mean()),        4) if len(conf_wrong)   else None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("Metrics saved to:", METRICS_JSON)
"""


def to_lines(src):
    """Convert a source string to the list-of-lines format Jupyter uses."""
    lines = src.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            if line:  # non-empty last line (no trailing newline)
                result.append(line)
    return result


nb = json.load(open(NB_PATH, encoding="utf-8"))

patches = {3: CELL3, 4: CELL4, 8: CELL8, 9: CELL9, 10: CELL10}
for idx, src in patches.items():
    nb["cells"][idx]["source"] = to_lines(src)
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook patched successfully.")
print("Modified cells: 3, 4, 8, 9, 10")
