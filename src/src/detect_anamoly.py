"""
detect_anamoly.py
─────────────────
Quick sanity test using the NEW model (ids_model.pkl + scaler.pkl).
Tests 9 scenarios matching the feature order used in training:
[cpu, mem, disk, net_kb_s, proc_count, spawn_rate, sudo, failed_logins, cpu_mem_ratio]
"""

import numpy as np, os, joblib

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model      = joblib.load(os.path.join(BASE_DIR, "models", "ids_model.pkl"))
scaler     = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
threshold  = float(open(os.path.join(BASE_DIR, "models", "threshold.txt")).read().strip())

# [cpu, mem, disk, net_KB/s, proc, spawn, sudo, fail]
test_cases = [
    ("Normal idle",        15,  82, 30,    5,  320,  0.0, 0,   0),
    ("Normal active",      35,  83, 30,   20,  330,  0.5, 0,   0),
    ("Normal heavy",       70,  85, 35,   50,  350,  2.0, 0,   0),
    ("CPU spike",          97,  85, 30,   15,  400, 12.0, 0,   0),
    ("Process bomb",       80,  84, 30,   10, 1200, 30.0, 0,   0),
    ("Data exfiltration",  20,  83, 30,  900,  330,  0.0, 0,   0),
    ("Brute force",        20,  83, 30,   10,  330,  0.0, 0, 150),
    ("High mem",           30,  97, 30,   10,  340,  0.5, 0,   0),
    ("Sudo flood",         25,  83, 30,   10,  330,  1.0, 80,  0),
]

print(f"\n{'CASE':<22} {'SCORE':>8}  {'VERDICT'}")
print("-" * 45)
for label, cpu, mem, disk, net, proc, spawn, sudo, fail in test_cases:
    cpu_mem = cpu / (mem + 1e-5)
    row     = np.array([[cpu, mem, disk, net, proc, spawn, sudo, fail, cpu_mem]])
    scaled  = scaler.transform(row)
    score   = model.decision_function(scaled)[0]
    verdict = "ANOMALY !" if score < threshold else "normal"
    print(f"{label:<22} {score:>8.4f}  {verdict}")

print(f"\nThreshold: {threshold:.4f}")
