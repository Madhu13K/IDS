import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "scaled_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.joblib")

# load model and training data
model = joblib.load(MODEL_PATH)
X_train = pd.read_csv(DATA_PATH).values

# get anomaly scores for training data
train_scores = model.decision_function(X_train)

# simulate test samples (same as detect file)
X_test = np.array([
    [20, 10, 70, 4000, 8000, 30, 25, 120],
    [95, 90, 5,  7900, 100,  900, 850, 2000],
    [25, 15, 60, 4200, 7800, 35, 30, 140],
    [98, 92, 3,  8000, 50,   950, 900, 2200]
])

# scale test data using the same scaler
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.joblib"))
X_test_scaled = scaler.transform(X_test)

test_scores = model.decision_function(X_test_scaled)

# plot both distributions
plt.hist(train_scores, bins=50, alpha=0.7, label="Training (Normal)")
plt.hist(test_scores, bins=20, alpha=0.7, label="Test Samples")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.title("Anomaly Score Distribution")
plt.legend()
plt.show()

# print stats
print("Training score range:", np.min(train_scores), "to", np.max(train_scores))
print("Test score range:", np.min(test_scores), "to", np.max(test_scores))

# choose threshold (5th percentile of training scores)
threshold = np.percentile(train_scores, 5)
print("Calibrated threshold:", threshold)

# save threshold for live detection
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.txt")

with open(THRESHOLD_PATH, "w") as f:
    f.write(str(threshold))

print("Threshold saved at:", THRESHOLD_PATH)

