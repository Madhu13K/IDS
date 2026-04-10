import os
import joblib
import numpy as np
import pandas as pd

# project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "ids_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.txt")

# -----------------------------
# LOAD MODEL + SCALER + THRESHOLD
# -----------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = float(f.read().strip())

# -----------------------------
# SIMULATED INPUT (MUST MATCH TRAINING)
# -----------------------------
cpu = 95
memory = 90
disk = 50
net_out = 900
process_count = 400
process_spawn_rate = 25
sudo_commands = 1
failed_logins = 0

# derived feature (VERY IMPORTANT)
cpu_memory_ratio = cpu / (memory + 1e-5)

# -----------------------------
# CREATE FEATURE VECTOR (9 FEATURES)
# -----------------------------
columns = [
    "cpu_usage",
    "memory_usage",
    "disk_io",
    "net_out",
    "process_count",
    "process_spawn_rate",
    "sudo_commands",
    "failed_logins",
    "cpu_memory_ratio"
]

new_sample = pd.DataFrame([[
    cpu,
    memory,
    disk,
    net_out,
    process_count,
    process_spawn_rate,
    sudo_commands,
    failed_logins,
    cpu / (memory + 1e-5)
]], columns=columns)

# -----------------------------
# SCALE
# -----------------------------
new_sample_scaled = scaler.transform(new_sample)

# -----------------------------
# PREDICT
# -----------------------------
score = model.decision_function(new_sample_scaled)[0]

# -----------------------------
# DECISION
# -----------------------------
if score < THRESHOLD:
    print(f"🚨 INTRUSION DETECTED | Score: {score:.4f}")
else:
    print(f"✅ NORMAL BEHAVIOR | Score: {score:.4f}")