import warnings
warnings.filterwarnings("ignore")
import os
import joblib
import numpy as np

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "ids_model.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "models", "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.txt")

model     = joblib.load(MODEL_PATH)
scaler    = joblib.load(SCALER_PATH)
threshold = float(open(THRESHOLD_PATH).read().strip())

def score_sample(cpu, mem, disk, net, proc, spawn, sudo, fail):
    cpu_memory_ratio = cpu / (mem + 1e-5)
    row = [[cpu, mem, disk, net, proc, spawn, sudo, fail, cpu_memory_ratio]]
    scaled = scaler.transform(row)
    return model.decision_function(scaled)[0]

cases = [
    ("Normal (idle)",       22,  40, 55,  180, 145,  3.1,  0,  0),
    ("Normal (active)",     45,  60, 70,  350, 180,  4.5,  1,  0),
    ("CPU spike",           92,  91, 50,  900, 400, 25.0,  1,  0),
    ("Brute-force login",   30,  45, 55,  200, 150,  3.0,  0, 150),
    ("Cryptominer",         97,  85, 40, 5000, 200,  8.0,  0,  0),
    ("Reverse shell",       35,  50, 55, 9500, 155,  3.2,  8,  5),
    ("Process bomb",        80,  70, 60,  250,1200,120.0,  0,  0),
]

print(f"\n{'LABEL':<22} {'SCORE':>8}  VERDICT")
print("-" * 45)
for label, *vals in cases:
    s = score_sample(*vals)
    verdict = "ANOMALY" if s < threshold else "NORMAL"
    print(f"  {label:<20} {s:>8.4f}  {verdict}")
print()