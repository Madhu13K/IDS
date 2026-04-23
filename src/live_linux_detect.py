import os
import joblib
import numpy as np
import psutil
import time
from datetime import datetime

# ── Path setup ─────────────────────────────────────────────────────────────────
# If this file is at project root:  BASE_DIR = dirname(__file__)
# If this file is inside src/:      BASE_DIR = dirname(dirname(__file__))
# Adjust the line below to match YOUR layout.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # ← project root
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ← if inside src/

MODEL_PATH     = os.path.join(BASE_DIR, "models", "ids_model.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "models", "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.txt")

# ── Load model ─────────────────────────────────────────────────────────────────
model     = joblib.load(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)
threshold = float(open(THRESHOLD_PATH).read().strip())
print(f"[IDS] Model loaded. Threshold: {threshold}")

# ── Baseline state for delta calculations ──────────────────────────────────────
_last_net        = psutil.net_io_counters().bytes_sent
_last_net_t      = datetime.now()
_last_proc_count = len(psutil.pids())
_last_proc_t     = datetime.now()


def collect_metrics():
    """
    Collect all 9 features that match training data exactly:
      cpu_usage, memory_usage, disk_io, net_out (KB/s delta),
      process_count, process_spawn_rate (new procs/sec),
      sudo_commands, failed_logins, cpu_memory_ratio
    """
    global _last_net, _last_net_t, _last_proc_count, _last_proc_t

    cpu    = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk   = psutil.disk_usage('/').percent

    # ── net_out: KB/s delta (NOT cumulative bytes) ─────────────────────────────
    now         = datetime.now()
    current_net = psutil.net_io_counters().bytes_sent
    elapsed_net = (now - _last_net_t).total_seconds() or 1
    net_out     = (current_net - _last_net) / 1024 / elapsed_net   # KB/s
    _last_net   = current_net
    _last_net_t = now

    # ── process_count and spawn_rate: new processes per second ─────────────────
    current_proc     = len(psutil.pids())
    elapsed_proc     = (now - _last_proc_t).total_seconds() or 1
    proc_delta       = max(0, current_proc - _last_proc_count)
    spawn_rate       = round(proc_delta / elapsed_proc, 3)
    _last_proc_count = current_proc
    _last_proc_t     = now

    # ── Linux: read failed logins and sudo from auth.log ───────────────────────
    # (returns 0 if log is unreadable — safe fallback)
    sudo_commands = 0
    failed_logins = 0

    try:
        auth_log = "/var/log/auth.log"
        if os.path.exists(auth_log):
            cutoff = time.time() - 60   # last 60 seconds
            with open(auth_log, "r", errors="ignore") as f:
                for line in f:
                    if "sudo:" in line:
                        sudo_commands += 1
                    if "Failed password" in line or "authentication failure" in line:
                        failed_logins += 1
    except PermissionError:
        pass   # if log isn't readable, stay at 0

    # ── derived feature (must match training) ──────────────────────────────────
    cpu_memory_ratio = cpu / (memory + 1e-5)

    return {
        "cpu":    cpu,
        "memory": memory,
        "disk":   disk,
        "net":    round(net_out, 2),
        "proc":   current_proc,
        "spawn":  spawn_rate,
        "sudo":   sudo_commands,
        "fail":   failed_logins,
        "ratio":  round(cpu_memory_ratio, 4),
    }


# ── Continuous monitoring loop ─────────────────────────────────────────────────
print("[IDS] Starting live monitoring. Press Ctrl+C to stop.\n")

while True:
    m = collect_metrics()

    features = [[
        m["cpu"],
        m["memory"],
        m["disk"],
        m["net"],
        m["proc"],
        m["spawn"],
        m["sudo"],
        m["fail"],
        m["ratio"],
    ]]

    scaled = scaler.transform(features)
    score  = model.decision_function(scaled)[0]

    verdict = "🚨 INTRUSION DETECTED" if score < threshold else "✅ NORMAL"
    ts      = datetime.now().strftime("%H:%M:%S")

    print(f"[{ts}] {verdict} | score={score:.4f} thresh={threshold:.4f}")
    print(f"         cpu={m['cpu']:.1f}% mem={m['memory']:.1f}% "
          f"net={m['net']:.1f}KB/s proc={m['proc']} "
          f"spawn={m['spawn']:.2f} sudo={m['sudo']} fail={m['fail']}")
    print()

    time.sleep(2)