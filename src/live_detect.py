import os
import joblib
import numpy as np
import pandas as pd

# project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.txt")

# load artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = float(f.read().strip())

# simulate new incoming system behavior (1 sample, 8 features)
new_sample = np.array([[45, 120, 300, 0.15, 80, 10, 0.02, 1]])

# scale using existing scaler
new_sample_scaled = scaler.transform(new_sample)

# compute anomaly score
score = model.score_samples(new_sample_scaled)[0]

# decision
if score < THRESHOLD:
    print(f"🚨 INTRUSION DETECTED | Score: {score:.4f}")
else:
    print(f"✅ NORMAL BEHAVIOR | Score: {score:.4f}")
