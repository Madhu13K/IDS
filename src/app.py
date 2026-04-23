import warnings
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request
from flask_cors import CORS
import psutil, joblib, os, numpy as np
from datetime import datetime, timedelta
import time

app = Flask(__name__)
CORS(app)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Layout B — app.py is inside src/ (models/ is one level up):
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Model loading ──────────────────────────────────────────────────────────────
model     = joblib.load(os.path.join(BASE_DIR, "models", "ids_model.pkl"))
scaler    = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
threshold = float(open(os.path.join(BASE_DIR, "models", "threshold.txt")).read().strip())

print(f"[IDS] Model loaded. Threshold: {threshold}")

# ── Baselines captured at startup ─────────────────────────────────────────────
_last_net        = psutil.net_io_counters().bytes_sent
_last_net_t      = datetime.now()
_last_proc_count = len(psutil.pids())
_last_proc_t     = datetime.now()


# ── Platform-aware sudo/fail-login helper ─────────────────────────────────────
def get_security_events():
    """
    Windows: reads Security Event Log (requires pywin32).
    Linux/Mac: reads /var/log/auth.log for the last 60 seconds.
    Returns (sudo_count, failed_login_count).
    """
    import platform
    if platform.system() == "Windows":
        return _windows_events()
    else:
        return _linux_events()


def _windows_events():
    try:
        import win32evtlog
        from datetime import timedelta

        def count_event(event_id):
            hand   = win32evtlog.OpenEventLog(None, "Security")
            flags  = (win32evtlog.EVENTLOG_BACKWARDS_READ |
                      win32evtlog.EVENTLOG_SEQUENTIAL_READ)
            cutoff = datetime.now() - timedelta(minutes=1)
            count  = 0
            while True:
                events = win32evtlog.ReadEventLog(hand, flags, 0)
                if not events:
                    break
                for ev in events:
                    ev_time = datetime(*ev.TimeGenerated.timetuple()[:6])
                    if ev_time < cutoff:
                        win32evtlog.CloseEventLog(hand)
                        return count
                    if ev.EventID & 0xFFFF == event_id:
                        count += 1
            win32evtlog.CloseEventLog(hand)
            return count

        sudo = count_event(4672)   # special privileges assigned
        fail = count_event(4625)   # failed logon
        return sudo, fail

    except ImportError:
        print("[WARN] pywin32 not installed — sudo/fail will be 0")
        return 0, 0
    except Exception as e:
        print(f"[WARN] Windows Event Log error: {e}")
        return 0, 0


def _linux_events():
    sudo, fail = 0, 0
    try:
        log_path = "/var/log/auth.log"
        if not os.path.exists(log_path):
            return 0, 0
        cutoff = time.time() - 60
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                if "sudo:" in line:
                    sudo += 1
                if "Failed password" in line or "authentication failure" in line:
                    fail += 1
    except PermissionError:
        pass
    return sudo, fail


# ── /metrics — real-time data from THIS machine ────────────────────────────────
@app.route("/metrics")
def metrics():
    global _last_net, _last_net_t, _last_proc_count, _last_proc_t

    cpu  = psutil.cpu_percent(interval=1)
    mem  = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    # net KB/s delta — NOT cumulative bytes
    now         = datetime.now()
    current_net = psutil.net_io_counters().bytes_sent
    elapsed_net = (now - _last_net_t).total_seconds() or 1
    net         = (current_net - _last_net) / 1024 / elapsed_net
    _last_net   = current_net
    _last_net_t = now

    # process spawn rate — new processes per second
    current_proc     = len(psutil.pids())
    elapsed_proc     = (now - _last_proc_t).total_seconds() or 1
    proc_delta       = max(0, current_proc - _last_proc_count)
    spawn            = round(proc_delta / elapsed_proc, 3)
    _last_proc_count = current_proc
    _last_proc_t     = now

    sudo, fail = get_security_events()

    cpu_memory_ratio = cpu / (mem + 1e-5)

    row    = [[cpu, mem, disk, net, current_proc, spawn, sudo, fail, cpu_memory_ratio]]
    scaled = scaler.transform(row)
    score  = model.decision_function(scaled)[0]
    intrusion = bool(score < threshold)

    status = "INTRUSION" if intrusion else "normal"
    print(f"[{now.strftime('%H:%M:%S')}] score={score:.4f} thresh={threshold:.4f} "
          f"cpu={cpu:.1f}% mem={mem:.1f}% net={net:.1f}KB/s "
          f"proc={current_proc} spawn={spawn:.2f} sudo={sudo} fail={fail} → {status}")

    return jsonify({
        "cpu": round(cpu, 1), "mem": round(mem, 1), "disk": round(disk, 1),
        "proc": current_proc,  "net": round(net, 1),
        "sudo": sudo,          "fail": fail,
        "score":     round(float(score), 4),
        "threshold": round(threshold,    4),
        "intrusion": intrusion,
    })


# ── /inject — score a manually entered sample from the dashboard ───────────────
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
          f"sudo={sudo} fail={fail} → score={score:.4f} {'INTRUSION' if intrusion else 'normal'}")

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