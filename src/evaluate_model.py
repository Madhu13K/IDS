"""
evaluate_model.py
─────────────────
Formal evaluation of the IDS Isolation Forest model.

Produces:
  - Confusion matrix
  - Precision, Recall, F1-score
  - ROC-AUC (on held-out test data — NOT training data)
  - Score distribution plot
  - reports/evaluation_report.txt

NOTE on anomaly data:
  The calibration dataset (anomaly_for_calibration.csv) is behavioral data
  from a second machine with a distinct usage profile, not labelled attack
  traffic. Isolation Forest is unsupervised — it learns normality from your
  machine's behavior and flags deviations. The ROC-AUC metric reflects
  separation between the two behavioral distributions.

Run from project root:
    python src/evaluate_model.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # no display needed — saves to file
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORMAL_PATH    = os.path.join(BASE_DIR, "data", "processed", "scaled_data.csv")
ANOMALY_PATH   = os.path.join(BASE_DIR, "data", "raw",       "anomaly_for_calibration.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "models",            "ids_model.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "models",            "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models",            "threshold.txt")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
print("\n[1/6] Loading model...")
model     = joblib.load(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)
threshold = float(open(THRESHOLD_PATH).read().strip())
print(f"      Threshold: {threshold}")

# ── Load normal data and create held-out test split ───────────────────────────
print("\n[2/6] Creating train/test split on normal data...")
normal_df = pd.read_csv(NORMAL_PATH)

# drop non-feature columns if present
for col in ["machine_id", "timestamp", "is_anomaly"]:
    if col in normal_df.columns:
        normal_df = normal_df.drop(columns=[col])

normal_df = normal_df.apply(pd.to_numeric, errors="coerce").dropna()
X_normal  = normal_df.values

# 80/20 split — model was trained on 80%, we evaluate on the held-out 20%
X_train_normal, X_test_normal = train_test_split(
    X_normal, test_size=0.20, random_state=42
)
print(f"      Normal total      : {len(X_normal)} rows")
print(f"      Training portion  : {len(X_train_normal)} rows  (not used here — info only)")
print(f"      Held-out test set : {len(X_test_normal)} rows  ← evaluation uses this")

# ── Load anomaly calibration data ─────────────────────────────────────────────
print("\n[3/6] Loading anomaly calibration data...")
anomaly_df = pd.read_csv(ANOMALY_PATH)

# drop non-feature columns
for col in ["machine_id", "timestamp", "is_anomaly", "scenario"]:
    if col in anomaly_df.columns:
        anomaly_df = anomaly_df.drop(columns=[col])

# align columns to match training features
expected_cols = list(normal_df.columns)
for col in expected_cols:
    if col not in anomaly_df.columns:
        anomaly_df[col] = 0.0          # fill missing features with 0

anomaly_df = anomaly_df[expected_cols]
anomaly_df = anomaly_df.apply(pd.to_numeric, errors="coerce").dropna()

# scale using the same scaler used during training
X_anomaly_scaled = scaler.transform(anomaly_df.values)

print(f"      Anomaly rows      : {len(anomaly_df)}")
print(f"      NOTE: This is behavioral data from a second machine used for")
print(f"      threshold calibration — not labelled attack traffic.")

# ── Build evaluation dataset ──────────────────────────────────────────────────
print("\n[4/6] Building evaluation dataset...")

# held-out normal rows → label 0 (normal)
# anomaly calibration rows → label 1 (anomaly / out-of-distribution)
X_eval = np.vstack([X_test_normal, X_anomaly_scaled])
y_true = np.array([0] * len(X_test_normal) + [1] * len(X_anomaly_scaled))

print(f"      Eval set size     : {len(X_eval)} rows")
print(f"        — normal (held-out) : {len(X_test_normal)}")
print(f"        — anomaly (calib)   : {len(X_anomaly_scaled)}")

# ── Score and predict ─────────────────────────────────────────────────────────
scores    = model.decision_function(X_eval)
# Isolation Forest: score < threshold → anomaly (label 1)
y_pred    = (scores < threshold).astype(int)

# ── Metrics ───────────────────────────────────────────────────────────────────
print("\n[5/6] Computing metrics...")

precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)

# ROC-AUC: flip scores because lower score = more anomalous
roc_auc   = roc_auc_score(y_true, -scores)

cm        = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

report_lines = [
    "=" * 60,
    "  IDS MODEL EVALUATION REPORT",
    "=" * 60,
    "",
    "  Dataset",
    f"    Normal (held-out 20%, never seen during training): {len(X_test_normal)}",
    f"    Anomaly (calibration data, second machine):        {len(X_anomaly_scaled)}",
    f"    Total evaluated:                                   {len(X_eval)}",
    "",
    "  NOTE: Anomaly data is behavioral data from a second machine",
    "  with a distinct usage profile, used for threshold calibration.",
    "  Isolation Forest is unsupervised — it detects deviations from",
    "  learned normal behavior, not specific attack signatures.",
    "",
    "  Confusion Matrix",
    f"    True  Negatives (correct normal)   : {tn}",
    f"    False Positives (false alarms)      : {fp}",
    f"    False Negatives (missed anomalies)  : {fn}",
    f"    True  Positives (caught anomalies)  : {tp}",
    "",
    "  Performance Metrics",
    f"    Precision : {precision:.4f}  (of alerts raised, how many were real)",
    f"    Recall    : {recall:.4f}  (of anomalies, how many were caught)",
    f"    F1 Score  : {f1:.4f}  (harmonic mean of precision and recall)",
    f"    ROC-AUC   : {roc_auc:.4f}  (separation between normal and anomaly distributions)",
    "",
    f"    Threshold used: {threshold}",
    "",
    "  Limitations",
    "    - Training data is from a single Windows machine (VS Code + Chrome workload)",
    "    - Anomaly data is a second machine's idle/active behavior, not attack data",
    "    - sudo_commands and failed_logins are 0 on Windows without pywin32",
    "    - process_spawn_rate is an approximation (new PIDs per second)",
    "    - Model may not generalise to server workloads or Linux environments",
    "",
    "=" * 60,
]

report_text = "\n".join(report_lines)
print(report_text)

# save report
report_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")
with open(report_path, "w") as f:
    f.write(report_text)
print(f"\n  Report saved → {report_path}")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n[6/6] Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("IDS Model Evaluation", fontsize=14, fontweight="bold")

# — Plot 1: Confusion Matrix —
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Normal", "Anomaly"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

# — Plot 2: Score Distribution —
normal_scores  = scores[:len(X_test_normal)]
anomaly_scores = scores[len(X_test_normal):]

axes[1].hist(normal_scores,  bins=40, alpha=0.7, color="#378ADD", label="Normal (held-out)")
axes[1].hist(anomaly_scores, bins=20, alpha=0.7, color="#E24B4A", label="Anomaly (calib)")
axes[1].axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
axes[1].set_xlabel("Anomaly Score")
axes[1].set_ylabel("Count")
axes[1].set_title("Score Distribution\n(normal vs anomaly calibration data)")
axes[1].legend(fontsize=8)

# — Plot 3: ROC Curve —
fpr_vals, tpr_vals, _ = roc_curve(y_true, -scores)
axes[2].plot(fpr_vals, tpr_vals, color="#378ADD", linewidth=2,
             label=f"ROC-AUC = {roc_auc:.4f}")
axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curve\n(normal vs anomaly distributions)")
axes[2].legend(fontsize=9)
axes[2].set_xlim([0, 1])
axes[2].set_ylim([0, 1.02])

plt.tight_layout()
plot_path = os.path.join(REPORTS_DIR, "evaluation_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plots saved  → {plot_path}")

print("\n" + "=" * 60)
print("  EVALUATION COMPLETE")
print("=" * 60)
print(f"  F1: {f1:.4f}  |  ROC-AUC: {roc_auc:.4f}  |  Recall: {recall:.4f}")
print(f"  Check reports/ folder for full output.")
print("=" * 60 + "\n")