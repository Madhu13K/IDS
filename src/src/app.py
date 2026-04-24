import warnings
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request
from flask_cors import CORS
import psutil, joblib, os, numpy as np, subprocess
from datetime import datetime
from collections import deque

app = Flask(__name__)
CORS(app)

# ── Model loading ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model     = joblib.load(os.path.join(BASE_DIR, "models", "ids_model.pkl"))
scaler    = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
threshold = float(open(os.path.join(BASE_DIR, "models", "threshold.txt")).read().strip())

# Global variable for dynamic threshold
dynamic_threshold = threshold      # start from the threshold read from file
score_window = deque(maxlen=100)   # stores last 100 scores
WARMUP = 20                        # wait before adapting

print(f"[IDS] Model loaded. Threshold: {threshold}")

# ── Baselines captured at startup ─────────────────────────────────────────────
_last_net        = psutil.net_io_counters().bytes_sent
_last_net_t      = datetime.now()
_last_proc_count = len(psutil.pids())
_last_proc_t     = datetime.now()


# ── Linux auth log helpers (replaces Windows Event Log) ───────────────────────
def get_linux_failed_logins():
    for log_path in ["/var/log/auth.log", "/var/log/secure"]:
        if os.path.exists(log_path):
            try:
                result = subprocess.run(
                    ["grep", "-c", "Failed password", log_path],
                    capture_output=True, text=True, timeout=2
                )
                return int(result.stdout.strip()) if result.returncode == 0 else 0
            except Exception:
                return 0
    return 0

def get_linux_sudo_count():
    for log_path in ["/var/log/auth.log", "/var/log/secure"]:
        if os.path.exists(log_path):
            try:
                result = subprocess.run(
                    ["grep", "-c", "sudo:", log_path],
                    capture_output=True, text=True, timeout=2
                )
                return int(result.stdout.strip()) if result.returncode == 0 else 0
            except Exception:
                return 0
    return 0


# ── /metrics ────────────────────────────────────────────────────────────────────
@app.route("/metrics")
def metrics():
    global _last_net, _last_net_t, _last_proc_count, _last_proc_t, dynamic_threshold

    cpu  = psutil.cpu_percent(interval=1)
    mem  = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    now         = datetime.now()
    current_net = psutil.net_io_counters().bytes_sent
    elapsed_net = (now - _last_net_t).total_seconds() or 1
    net         = (current_net - _last_net) / 1024 / elapsed_net
    _last_net   = current_net
    _last_net_t = now

    current_proc     = len(psutil.pids())
    elapsed_proc     = (now - _last_proc_t).total_seconds() or 1
    proc_delta       = max(0, current_proc - _last_proc_count)
    spawn            = round(proc_delta / elapsed_proc, 3)
    _last_proc_count = current_proc
    _last_proc_t     = now

    sudo = get_linux_sudo_count()
    fail = get_linux_failed_logins()

    cpu_memory_ratio = cpu / (mem + 1e-5)

    row    = [[cpu, mem, disk, net, current_proc, spawn, sudo, fail, cpu_memory_ratio]]
    scaled = scaler.transform(row)
    score = model.decision_function(scaled)[0]

    # store score
    score_window.append(score)

    # adaptive threshold
    if len(score_window) >= WARMUP:
        new_thresh = np.percentile(score_window, 10)
        dynamic_threshold = 0.8 * dynamic_threshold + 0.2 * new_thresh  # smooth it

    intrusion = bool(score < dynamic_threshold)

    status = "INTRUSION" if intrusion else "normal"
    print(f"[{now.strftime('%H:%M:%S')}] score={score:.4f} thresh={dynamic_threshold:.4f} "
          f"cpu={cpu:.1f}% mem={mem:.1f}% net={net:.1f}KB/s "
          f"proc={current_proc} spawn={spawn:.2f} sudo={sudo} fail={fail} -> {status}")

    return jsonify({
        "cpu": round(cpu, 1), "mem": round(mem, 1), "disk": round(disk, 1),
        "proc": current_proc,  "net": round(net, 1),
        "sudo": sudo,          "fail": fail,
        "score":     round(float(score), 4),
        "threshold": round(dynamic_threshold, 4),
        "intrusion": intrusion,
    })


# ── /inject ────────────────────────────────────────────────────────────────────
@app.route("/inject")
def inject():
    cpu   = float(request.args.get("cpu",   30))
    mem   = float(request.args.get("mem",   45))
    disk  = float(request.args.get("disk",  55))
    net   = float(request.args.get("net",   10))
    proc  = float(request.args.get("proc",  330))
    spawn = float(request.args.get("spawn", 0))
    sudo  = float(request.args.get("sudo",  0))
    fail  = float(request.args.get("fail",  0))

    cpu_memory_ratio = cpu / (mem + 1e-5)
    row    = [[cpu, mem, disk, net, proc, spawn, sudo, fail, cpu_memory_ratio]]
    scaled = scaler.transform(row)
    score  = model.decision_function(scaled)[0]
    intrusion = bool(score < threshold)

    print(f"[INJECT] cpu={cpu} mem={mem} net={net} proc={proc} "
          f"sudo={sudo} fail={fail} -> score={score:.4f} {'INTRUSION' if intrusion else 'normal'}")

    return jsonify({
        "cpu": cpu, "mem": mem, "disk": disk,
        "proc": int(proc), "net": net,
        "sudo": int(sudo), "fail": int(fail),
        "score":     round(float(score), 4),
        "threshold": round(threshold,    4),
        "intrusion": intrusion,
    })


if __name__ == "__main__":
    print("[IDS] Starting on http://127.0.0.1:5000")
    print("[IDS] Routes: /metrics (live) | /inject (manual)")
    print("[IDS] Press Ctrl+C to stop\n")
    app.run(port=5000, debug=False)