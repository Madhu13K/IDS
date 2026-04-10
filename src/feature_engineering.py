import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

# locate project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "normal_behavior.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "scaled_data.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# -----------------------------
# STEP 1: Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------
# STEP 2: Clean dataset (VERY IMPORTANT)
# -----------------------------

# remove duplicate header rows if present
df = df[df["cpu_usage"] != "cpu_usage"]

# convert everything to float
df = df.astype(float)

# -----------------------------
# STEP 3: Feature Engineering (small improvement)
# -----------------------------
df["cpu_memory_ratio"] = df["cpu_usage"] / (df["memory_usage"] + 1e-5)

# -----------------------------
# STEP 4: Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -----------------------------
# STEP 5: Save processed data
# -----------------------------
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

scaled_df = pd.DataFrame(X_scaled, columns=df.columns)
scaled_df.to_csv(PROCESSED_PATH, index=False)

# -----------------------------
# STEP 6: Save scaler
# -----------------------------
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
joblib.dump(scaler, SCALER_PATH)

# -----------------------------
# DONE
# -----------------------------
print("✅ Feature engineering + scaling complete.")
print("Shape:", X_scaled.shape)
print("Saved at:", PROCESSED_PATH)
print("Scaler saved at:", SCALER_PATH)