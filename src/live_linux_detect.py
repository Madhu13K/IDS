import os
import joblib
import numpy as np
import psutil
import time

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "ids_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.txt")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read().strip())

# -----------------------------
# STEP 1: COLLECT METRICS
# -----------------------------
def collect_metrics():
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    process_count = len(psutil.pids())

    net = psutil.net_io_counters()
    net_out = net.bytes_sent

    # simple approximations
    process_spawn_rate = process_count / 50
    sudo_commands = 0
    failed_logins = 0

    return [
        cpu,
        memory,
        disk,
        net_out,
        process_count,
        process_spawn_rate,
        sudo_commands,
        failed_logins
    ]

# -----------------------------
# STEP 2: CONTINUOUS MONITORING
# -----------------------------
while True:

    data = collect_metrics()

    cpu = data[0]
    memory = data[1]
    disk = data[2]
    net_out = data[3]
    process_count = data[4]
    process_spawn_rate = data[5]
    sudo_commands = data[6]
    failed_logins = data[7]

    # derived feature
    cpu_memory_ratio = cpu / (memory + 1e-5)

    # create input
    features = [[
        cpu,
        memory,
        disk,
        net_out,
        process_count,
        process_spawn_rate,
        sudo_commands,
        failed_logins,
        cpu_memory_ratio
    ]]

    # scale
    scaled = scaler.transform(features)

    # predict
    score = model.decision_function(scaled)[0]

    # display
    print("=" * 40)
    print(f"CPU: {cpu:.2f}% | Memory: {memory:.2f}%")
    
    if score < threshold:
        print(f"🚨 INTRUSION DETECTED | Score: {score:.4f}")
    else:
        print(f"✅ NORMAL | Score: {score:.4f}")

    print("=" * 40)

    time.sleep(2)