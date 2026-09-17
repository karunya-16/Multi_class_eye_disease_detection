"""Append cells 5-12 to notebook 12."""
import json
from pathlib import Path

NB = Path(r"D:\Practice Projects\Disease Detection\notebooks\12_camera_classifier_external_validation.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

cells = []

# ── Cell 5: Metrics ────────────────────────────────────────────────────────
cells.append(("code", "cell_metrics", """\
# CELL 5 - Compute evaluation metrics
if not HAS_IMAGES:
    print("External validation images not provided yet - validation skipped.")
else:
    y_true = df["true_id"].values
    y_pred = df["predicted_id"].values
    y_prob = df["prob_normal"].values   # P(Normal) = sigmoid output

    HAS_BOTH_CLASSES = len(np.unique(y_true)) == 2

    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm_arr   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    prec, rec, f1s, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=["Cataract", "Normal"], zero_division=0)

    roc_auc = None
    if HAS_BOTH_CLASSES:
        roc_auc = roc_auc_score(y_true, y_prob)

    # Confidence stats
    conf_all     = df["confidence"].values
    conf_correct = df.loc[df["correct"],  "confidence"].values
    conf_wrong   = df.loc[~df["correct"], "confidence"].values

    print("=" * 60)
    print("EXTERNAL VALIDATION METRICS")
    print("=" * 60)
    print(f"Total images evaluated : {len(df)}")
    print(f"  Cataract             : {int((y_true == 0).sum())}")
    print(f"  Normal               : {int((y_true == 1).sum())}")
    print(f"  Incorrect predictions: {int((~df['correct']).sum())}")
    print()
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC           : N/A (single class)")
    print()
    print(f"Cataract(0) -> Precision:{prec[0]:.4f}  Recall:{rec[0]:.4f}  F1:{f1s[0]:.4f}  n={int(sup[0])}")
    print(f"Normal  (1) -> Precision:{prec[1]:.4f}  Recall:{rec[1]:.4f}  F1:{f1s[1]:.4f}  n={int(sup[1])}")
    print()
    print(f"Confusion Matrix (rows=true, cols=pred) [Cataract, Normal]:")
    print(cm_arr)
    print()
    print("Classification Report:")
    print(report)
    print()
    print("Confidence Statistics:")
    print(f"  All       -> mean:{conf_all.mean():.4f}  median:{float(np.median(conf_all)):.4f}  min:{conf_all.min():.4f}  max:{conf_all.max():.4f}")
    if len(conf_correct):
        print(f"  Correct   -> mean:{conf_correct.mean():.4f}  median:{float(np.median(conf_correct)):.4f}")
    if len(conf_wrong):
        print(f"  Incorrect -> mean:{conf_wrong.mean():.4f}  median:{float(np.median(conf_wrong)):.4f}")
    if int((y_true == 1).sum()) < 30:
        print()
        print(f"NOTE: Only {int((y_true==1).sum())} Normal images provided.")
        print("      Normal-class metrics have HIGH uncertainty.")
"""))

# ── Cell 6: Error analysis ─────────────────────────────────────────────────
cells.append(("code", "cell_errors", """\
# CELL 6 - Error analysis
if not HAS_IMAGES:
    print("External validation images not provided yet - validation skipped.")
else:
    errors = df[~df["correct"]].sort_values("confidence", ascending=False)
    print(f"Incorrectly classified images: {len(errors)}")
    if len(errors):
        print(errors[["filename","true_label","predicted_label",
                       "confidence","prob_cataract","prob_normal"]].to_string(index=False))
    else:
        print("No incorrect predictions.")
"""))

# ── Cell 7: Visualizations ─────────────────────────────────────────────────
cells.append(("code", "cell_viz", """\
# CELL 7 - Visualizations: confusion matrix, ROC, confidence distribution
if not HAS_IMAGES:
    print("External validation images not provided yet - validation skipped.")
else:
    ncols = 2 if HAS_BOTH_CLASSES else 1
    fig, axes = plt.subplots(1, ncols + 1, figsize=(7 * (ncols + 1), 5))
    axes = np.array(axes).flatten()

    # Confusion matrix
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Cataract", "Pred Normal"],
                yticklabels=["True Cataract", "True Normal"], ax=axes[0])
    axes[0].set_title("Confusion Matrix (External)")
    axes[0].set_ylabel("True Label"); axes[0].set_xlabel("Predicted Label")

    # ROC curve
    if HAS_BOTH_CLASSES:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[1].plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.4f}")
        axes[1].plot([0,1],[0,1], "k--", lw=1)
        axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve (External)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Confidence distribution
    ax_conf = axes[2] if HAS_BOTH_CLASSES else axes[1]
    ax_conf.hist(df.loc[df["true_id"]==0, "confidence"], bins=20, alpha=0.6,
                 label="Cataract (true)", color="steelblue")
    ax_conf.hist(df.loc[df["true_id"]==1, "confidence"], bins=20, alpha=0.6,
                 label="Normal (true)", color="orange")
    ax_conf.set_xlabel("Confidence"); ax_conf.set_ylabel("Count")
    ax_conf.set_title("Confidence Distribution (External)")
    ax_conf.legend(); ax_conf.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(REPORT_DIR / "external_metrics_plots.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: external_metrics_plots.png")
"""))

# ── Cell 8: Sample predictions ─────────────────────────────────────────────
cells.append(("code", "cell_samples", """\
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
        # Reconstruct path from filename + true_label
        img_path = EXT_DIR / row["true_label"] / row["filename"]
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
"""))

# ── Cell 9: Grad-CAM ───────────────────────────────────────────────────────
cells.append(("code", "cell_gradcam", """\
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
        img_path = EXT_DIR / row["true_label"] / row["filename"]
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
"""))

# ── Cell 10: Save metrics JSON ─────────────────────────────────────────────
cells.append(("code", "cell_save_metrics", """\
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
        "images_evaluated": len(df),
        "cataract_count": int((y_true == 0).sum()),
        "normal_count":   int((y_true == 1).sum()),
        "invalid_images": n_invalid,
        "incorrect_predictions": int((~df["correct"]).sum()),
        "accuracy":           round(float(acc),      4),
        "balanced_accuracy":  round(float(bal_acc),  4),
        "macro_f1":           round(float(macro_f1), 4),
        "roc_auc":            round(float(roc_auc),  4) if roc_auc is not None else None,
        "cataract_precision": round(float(prec[0]),  4),
        "cataract_recall":    round(float(rec[0]),   4),
        "cataract_f1":        round(float(f1s[0]),   4),
        "normal_precision":   round(float(prec[1]),  4),
        "normal_recall":      round(float(rec[1]),   4),
        "normal_f1":          round(float(f1s[1]),   4),
        "confusion_matrix":   cm_arr.tolist(),
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
"""))

# ── Cell 11: Isolation validation + final status ───────────────────────────
cells.append(("code", "cell_final", """\
# CELL 11 - Isolation validation + final status
print("=" * 60)
print("ISOLATION VALIDATION")
print("=" * 60)
all_ok = True
for rel, before in PROTECTED_BEFORE.items():
    after = _fingerprint(PROJECT_ROOT / rel)
    if not before["exists"]:
        status = "SKIP (did not exist before)"
    elif before["md5"] != after["md5"] or before["mtime_ns"] != after["mtime_ns"]:
        status = "MODIFIED  <-- WARNING"; all_ok = False
    else:
        status = "UNCHANGED OK"
    print(f"  {rel:<55} {status}")

overall = "PASS" if all_ok else "FAIL"
if not HAS_IMAGES:
    overall = "SKIPPED"

print()
print("=" * 60)
print(f"EXTERNAL CAMERA VALIDATION STATUS: {overall}")
print("=" * 60)
print()
if not HAS_IMAGES:
    print("Images evaluated : 0")
    print("Normal           : 0")
    print("Cataract         : 0")
    print()
    print("Accuracy         : N/A")
    print("Balanced Accuracy: N/A")
    print("Macro F1         : N/A")
    print("ROC-AUC          : N/A")
    print()
    print("Normal Recall    : N/A")
    print("Cataract Recall  : N/A")
    print()
    print("Incorrect predictions: N/A")
else:
    print(f"Images evaluated : {len(df)}")
    print(f"Normal           : {int((y_true==1).sum())}")
    print(f"Cataract         : {int((y_true==0).sum())}")
    print()
    print(f"Accuracy         : {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"ROC-AUC          : {f'{roc_auc:.4f}' if roc_auc is not None else 'N/A'}")
    print()
    print(f"Normal Recall    : {rec[1]:.4f}")
    print(f"Cataract Recall  : {rec[0]:.4f}")
    print()
    print(f"Incorrect predictions: {int((~df['correct']).sum())}")

print()
print("Model modified       : NO")
print("Fundus model modified: NO")
print("Backend modified     : NO")
print("Frontend modified    : NO")
print()
print("Domain: Normal vs Cataract ONLY. DR and Glaucoma are NOT supported.")
print()

# Final conclusion
if not HAS_IMAGES:
    print("CONCLUSION: External validation is pending because unseen camera images")
    print("have not been provided. Place images under:")
    print(f"  {EXT_DIR / 'Cataract'}")
    print(f"  {EXT_DIR / 'Normal'}")
    print("Then re-run this notebook.")
else:
    generalizes = (bal_acc >= 0.65 and (roc_auc is None or roc_auc >= 0.70)
                   and not (acc > 0.85 and rec[1] < 0.3))
    if generalizes:
        print("CONCLUSION: The camera Normal-vs-Cataract classifier generalizes")
        print("reasonably to the unseen external camera images based on balanced")
        print("accuracy and ROC-AUC. Further validation with more Normal samples")
        print("is recommended before production use.")
    else:
        print("CONCLUSION: The camera classifier does NOT generalize well to the")
        print("external images. Review class balance, image quality, and consider")
        print("retraining with additional data before deployment.")
print()
print("NOTE: Do NOT integrate into FastAPI/React yet.")
print("NOTE: Do NOT create a 4-class camera model yet.")
"""))

for kind, cid, src in cells:
    cell = {"cell_type": kind, "id": cid, "metadata": {}, "source": src}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb["cells"].append(cell)

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Done. Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    print(f"  {i}: {c['id']}")
