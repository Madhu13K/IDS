import os
import pandas as pd
import numpy as np

np.random.seed(42)

samples = 2000

data = {
    "cpu_usage": np.random.normal(30, 5, samples),
    "memory_usage": np.random.normal(45, 7, samples),
    "disk_io": np.random.normal(100, 20, samples),
    "net_out": np.random.normal(200, 50, samples),
    "process_count": np.random.normal(150, 20, samples),
    "process_spawn_rate": np.random.normal(3, 1, samples),
    "sudo_commands": np.random.poisson(1, samples),
    "failed_logins": np.random.poisson(0.2, samples)
}

df = pd.DataFrame(data)
df[df < 0] = 0

# 🔑 ABSOLUTE PATH FIX
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "normal_behavior.csv")

df.to_csv(DATA_PATH, index=False)

print("Dataset created at:", DATA_PATH)
