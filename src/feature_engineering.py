import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

# locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "normal_behavior.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "scaled_data.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

# load dataset
df = pd.read_csv(DATA_PATH)

# features only (no labels)
X = df.values

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 NEW: make sure processed folder exists
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

# 🔹 NEW: save scaled dataset
pd.DataFrame(X_scaled).to_csv(PROCESSED_PATH, index=False)

# save scaler
joblib.dump(scaler, SCALER_PATH)

print("Feature scaling complete.")
print("Scaled data shape:", X_scaled.shape)
print("Scaled data saved at:", PROCESSED_PATH)
print("Scaler saved at:", SCALER_PATH)
