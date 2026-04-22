"""
train_model.py
──────────────
Trains Isolation Forest on your real normal data.
Calibrates threshold statistically using friend's anomaly data.
No more hardcoded -0.12.

Run from project root:
    python src/train_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(BASE_DIR, "data", "processed", "scaled_data.csv")
ANOMALY_PATH   = os.path.join(BASE_DIR, "data", "raw", "anomaly_for_calibration.csv")
SCALER_PATH    = os.path.join(BASE_DIR, "models", "scaler.pkl")
MODEL_PATH     = os.path.join(BASE_DIR, "models", "ids_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.txt")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load scaled normal training data
# ════════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Loading scaled training data...")
df = pd.read_csv(DATA_PATH)

# drop machine_id if it snuck in
if "machine_id" in df.columns:
    df = df.drop(columns=["machine_id"])

X_train = df.values
print(f"    Shape: {X_train.shape}")
print(f"    Features: {list(df.columns)}")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — Train Isolation Forest
# ════════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Training Isolation Forest...")
model = IsolationForest(
    n_estimators=300,       # more trees = more stable scores
    contamination=0.005,    # ~0.5% of training data expected to be borderline
    max_samples="auto",
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
model.fit(X_train)

train_scores = model.decision_function(X_train)
print(f"    Training score range: {train_scores.min():.4f} to {train_scores.max():.4f}")
print(f"    Mean: {train_scores.mean():.4f}  Std: {train_scores.std():.4f}")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — Calibrate threshold using anomaly data
# ════════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Calibrating threshold...")

scaler = joblib.load(SCALER_PATH)

if not os.path.exists(ANOMALY_PATH):
    # fallback: use 2nd percentile of training scores
    threshold = float(np.percentile(train_scores, 2))
    print(f"    [WARN] No anomaly data found — using 2nd percentile of training scores")
    print(f"    Fallback threshold: {threshold:.4f}")
else:
    anomaly_df = pd.read_csv(ANOMALY_PATH)

    # drop non-feature columns
    drop_cols = ["machine_id", "timestamp", "is_anomaly", "scenario"]
    anomaly_df = anomaly_df.drop(columns=[c for c in drop_cols if c in anomaly_df.columns])

    # make sure columns match training
    anomaly_df = anomaly_df[df.columns]
    anomaly_df = anomaly_df.apply(pd.to_numeric, errors="coerce").dropna()

    X_anomaly_scaled = scaler.transform(anomaly_df.values)
    anomaly_scores   = model.decision_function(X_anomaly_scaled)

    print(f"    Anomaly score range : {anomaly_scores.min():.4f} to {anomaly_scores.max():.4f}")
    print(f"    Anomaly mean score  : {anomaly_scores.mean():.4f}")
    print(f"    Normal score range  : {train_scores.min():.4f} to {train_scores.max():.4f}")
    print(f"    Normal mean score   : {train_scores.mean():.4f}")

    # threshold = midpoint between:
    #   worst normal score (5th percentile of normals)
    #   best anomaly score (95th percentile of anomalies)
    # This gives a balanced boundary between the two distributions
    normal_p5  = np.percentile(train_scores, 5)
    anomaly_p95 = np.percentile(anomaly_scores, 95)

    if anomaly_p95 < normal_p5:
        # clean separation — set threshold halfway between them
        threshold = float((normal_p5 + anomaly_p95) / 2)
        print(f"\n    Clean separation detected:")
        print(f"      Normal  5th pct  : {normal_p5:.4f}")
        print(f"      Anomaly 95th pct : {anomaly_p95:.4f}")
        print(f"      Midpoint threshold: {threshold:.4f}")
    else:
        # distributions overlap — use normal 5th percentile as safe fallback
        threshold = float(normal_p5)
        print(f"\n    [WARN] Score distributions overlap — using normal 5th percentile")
        print(f"      Normal  5th pct  : {normal_p5:.4f}")
        print(f"      Anomaly 95th pct : {anomaly_p95:.4f}")
        print(f"      Threshold set to : {threshold:.4f}")

    # sanity check — what % of anomalies does this threshold catch?
    caught    = (anomaly_scores < threshold).sum()
    catch_rate = caught / len(anomaly_scores) * 100
    fpr_train  = (train_scores < threshold).sum() / len(train_scores) * 100
    print(f"\n    Anomaly detection rate : {caught}/{len(anomaly_scores)} = {catch_rate:.1f}%")
    print(f"    False positive rate    : {fpr_train:.1f}% on training data")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — Save model and threshold
# ════════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Saving model and threshold...")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

with open(THRESHOLD_PATH, "w") as f:
    f.write(str(round(threshold, 4)))

print(f"    Model saved     → {MODEL_PATH}")
print(f"    Threshold saved → {THRESHOLD_PATH}  (value: {threshold:.4f})")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 5 — Final sanity check
# ════════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Final sanity check on known cases...")

sanity_cases = [
    ("Normal idle",       15,  82, 30,   5,  320,  0.0, 0,   0),
    ("Normal active",     35,  83, 30,  20,  330,  0.5, 0,   0),
    ("Normal heavy",      70,  85, 35,  50,  350,  2.0, 0,   0),
    ("CPU spike",         97,  85, 30,  15,  400, 12.0, 0,   0),
    ("Process bomb",      80,  84, 30,  10, 1200, 30.0, 0,   0),
    ("Data exfiltration", 20,  83, 30, 900,  330,  0.0, 0,   0),
    ("Brute force",       20,  83, 30,  10,  330,  0.0, 0, 150),
]

print(f"\n    {'CASE':<22} {'SCORE':>8}  {'VERDICT':<12}")
print(f"    {'-'*45}")
for label, cpu, mem, disk, net, proc, spawn, sudo, fail in sanity_cases:
    cpu_mem_ratio = cpu / (mem + 1e-5)
    row = np.array([[cpu, mem, disk, net, proc, spawn, sudo, fail, cpu_mem_ratio]])
    scaled = scaler.transform(row)
    score  = model.decision_function(scaled)[0]
    verdict = "ANOMALY !" if score < threshold else "normal"
    print(f"    {label:<22} {score:>8.4f}  {verdict}")


print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  Model    : IsolationForest (300 trees)")
print(f"  Trained  : {len(X_train)} real normal rows")
print(f"  Threshold: {threshold:.4f} (statistically calibrated)")
print(f"\n  Restart Flask and open the dashboard.")
print(f"  python src/app.py")
print("=" * 60)