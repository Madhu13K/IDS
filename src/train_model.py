import pandas as pd
import os
import joblib
from sklearn.ensemble import IsolationForest

# locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "normal_behavior.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.joblib")

# load data
df = pd.read_csv(DATA_PATH)

# load scaler and scale data
scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(df.values)

# train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

# save model
joblib.dump(model, MODEL_PATH)

print("Isolation Forest model trained successfully.")
print("Model saved at:", MODEL_PATH)
