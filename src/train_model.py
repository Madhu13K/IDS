import pandas as pd
import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

# locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "scaled_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ids_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.txt")

# -----------------------------
# STEP 1: Load PROCESSED data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# STEP 2: Train model
# -----------------------------
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(df)

# -----------------------------
# STEP 3: Compute anomaly scores
# -----------------------------
scores = model.decision_function(df)

print("Training score range:", np.min(scores), "to", np.max(scores))

# -----------------------------
# STEP 4: Set threshold (IMPORTANT)
# -----------------------------
threshold = np.percentile(scores, 5)

# -----------------------------
# STEP 5: Save model + threshold
# -----------------------------
joblib.dump(model, MODEL_PATH)

with open(THRESHOLD_PATH, "w") as f:
    f.write(str(threshold))

# -----------------------------
# DONE
# -----------------------------
print("✅ Model trained successfully.")
print("Threshold:", threshold)
print("Model saved at:", MODEL_PATH)
print("Threshold saved at:", THRESHOLD_PATH)