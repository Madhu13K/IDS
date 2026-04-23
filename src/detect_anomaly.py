import pandas as pd
import os
import joblib

# locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

# load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# sample NEW behavior (simulated)
test_data = pd.DataFrame([
    [20, 10, 70, 4000, 8000, 30, 25, 120],   # normal
    [95, 90, 5,  7900, 100,  900, 850, 2000],# anomaly
    [25, 15, 60, 4200, 7800, 35, 30, 140],   # normal
    [98, 92, 3,  8000, 50,   950, 900, 2200] # anomaly
])


# scale data
X_scaled = scaler.transform(test_data.values)

# predict
predictions = model.predict(X_scaled)
scores = model.decision_function(X_scaled)

# display results
for i in range(len(predictions)):
    status = "NORMAL" if predictions[i] == 1 else "ANOMALY"
    print(f"Sample {i+1}: {status} | Score: {scores[i]:.4f}")
