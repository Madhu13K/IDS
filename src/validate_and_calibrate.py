"""
validate_and_calibrate.py
──────────────────────────
Visualises the score distribution of your trained model.
Uses ids_model.pkl + scaler.pkl (the new stack).
Run from project root: python src/validate_and_calibrate.py
"""

import pandas as pd, numpy as np, os, joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe on headless Linux)
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "scaled_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ids_model.pkl")
SCALER_PATH= os.path.join(BASE_DIR, "models", "scaler.pkl")
THRESH_PATH= os.path.join(BASE_DIR, "models", "threshold.txt")
PLOT_PATH  = os.path.join(BASE_DIR, "models", "score_distribution.png")

model     = joblib.load(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)
threshold = float(open(THRESH_PATH).read().strip())

df = pd.read_csv(DATA_PATH)
if "machine_id" in df.columns:
    df = df.drop(columns=["machine_id"])
X_train = df.values

train_scores = model.decision_function(X_train)

# Known anomaly test vectors [cpu,mem,disk,net_KB/s,proc,spawn,sudo,fail]
test_cases = np.array([
    [97,  85, 30,  15,  400, 12.0, 0,   0],   # CPU spike
    [80,  84, 30,  10, 1200, 30.0, 0,   0],   # process bomb
    [20,  83, 30, 900,  330,  0.0, 0,   0],   # exfiltration
    [20,  83, 30,  10,  330,  0.0, 0, 150],   # brute force
])
test_labels = ["CPU spike", "Proc bomb", "Exfiltration", "Brute force"]

# add cpu_memory_ratio
cpu_mem = (test_cases[:, 0] / (test_cases[:, 1] + 1e-5)).reshape(-1, 1)
test_cases_full = np.hstack([test_cases, cpu_mem])
test_scaled  = scaler.transform(test_cases_full)
test_scores  = model.decision_function(test_scaled)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(train_scores, bins=60, alpha=0.6, color="steelblue", label="Normal training data")
for score, label in zip(test_scores, test_labels):
    ax.axvline(score, color="red", linewidth=1.5, linestyle="--", label=f"{label} ({score:.3f})")
ax.axvline(threshold, color="black", linewidth=2, label=f"Threshold ({threshold:.4f})")
ax.set_xlabel("Anomaly score (lower = more anomalous)")
ax.set_ylabel("Count")
ax.set_title("Score distribution — normal data vs. injected anomalies")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"Plot saved to {PLOT_PATH}")

print(f"\nTraining scores — min: {train_scores.min():.4f}  max: {train_scores.max():.4f}  mean: {train_scores.mean():.4f}")
print(f"Threshold: {threshold:.4f}")
print(f"\n{'CASE':<18} {'SCORE':>8}  {'VERDICT'}")
print("-" * 36)
for score, label in zip(test_scores, test_labels):
    print(f"{label:<18} {score:>8.4f}  {'ANOMALY' if score < threshold else 'normal'}")
