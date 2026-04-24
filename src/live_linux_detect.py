"""
live_linux_detect.py
──────────────────────
Terminal-based live detector. Runs independently of Flask.
Uses ids_model.pkl + scaler.pkl (same as app.py).
net_out computed as KB/s delta (matches training data format).
"""

import os, joblib, numpy as np, psutil, time
from datetime import datetime

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "ids_model.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "models", "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.txt")

model     = joblib.load(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)
threshold = float(open(THRESHOLD_PATH).read().strip())

print(f"[IDS] Model loaded | Threshold: {threshold}")

# ── Delta tracking (must persist between loops) ────────────────────────────────
_last_net        = psutil.net_io_counters().bytes_sent
_last_net_t      = datetime.now()
_last_proc_count = len(psutil.pids())
_last_proc_t     = datetime.now()


def collect_metrics():
    global _last_net, _last_net_t, _last_proc_count, _last_proc_t

    cpu    = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk   = psutil.disk_usage("/").percent

    # net_out as KB/s delta (matches how training data was collected)
    now         = datetime.now()
    current_net = psutil.net_io_counters().bytes_sent
    elapsed_net = (now - _last_net_t).total_seconds() or 1
    net_out     = (current_net - _last_net) / 1024 / elapsed_net
    _last_net   = current_net
    _last_net_t = now

    # process spawn rate — new processes per second
    current_proc     = len(psutil.pids())
    elapsed_proc     = (now - _last_proc_t).total_seconds() or 1
    proc_delta       = max(0, current_proc - _last_proc_count)
    spawn            = round(proc_delta / elapsed_proc, 3)
    _last_proc_count = current_proc
    _last_proc_t     = now

    sudo_commands = 0   # always 0 (matches training data)
    failed_logins = 0

    cpu_memory_ratio = cpu / (memory + 1e-5)

    return cpu, memory, disk, net_out, current_proc, spawn, sudo_commands, failed_logins, cpu_memory_ratio


while True:
    cpu, memory, disk, net_out, proc_count, spawn, sudo, fail, cpu_mem_ratio = collect_metrics()

    features = [[cpu, memory, disk, net_out, proc_count, spawn, sudo, fail, cpu_mem_ratio]]
    scaled   = scaler.transform(features)
    score    = model.decision_function(scaled)[0]

    print("=" * 50)
    print(f"CPU: {cpu:.1f}%  MEM: {memory:.1f}%  NET: {net_out:.1f} KB/s")
    print(f"PROC: {proc_count}  SPAWN: {spawn:.2f}/s")

    if score < threshold:
        print(f"INTRUSION DETECTED | Score: {score:.4f} (thresh: {threshold:.4f})")
    else:
        print(f"NORMAL             | Score: {score:.4f} (thresh: {threshold:.4f})")

    print("=" * 50)
    time.sleep(2)
