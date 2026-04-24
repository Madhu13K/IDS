"""
merge_datasets.py
─────────────────
Merges your real normal data into combined_behavior.csv.
Friend's normal data dropped (idle machine, not representative).
Friend's anomaly data used only for threshold calibration.

Run from project root:
    python src/merge_datasets.py
"""

import os
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MY_DATA        = os.path.join(BASE_DIR, "src", "data", "raw", "my_normal_behavior.csv")
FRIEND_ANOMALY = os.path.join(BASE_DIR, "src", "data", "raw", "friend_anamoly.csv")

OUT_COMBINED   = os.path.join(BASE_DIR, "data", "raw", "combined_behavior.csv")
OUT_ANOMALY    = os.path.join(BASE_DIR, "data", "raw", "anomaly_for_calibration.csv")

STANDARD_COLS = [
    "cpu_usage", "memory_usage", "disk_io", "net_out",
    "process_count", "process_spawn_rate", "sudo_commands", "failed_logins"
]


# ── Helper: fix cumulative net_out → KB/s delta ────────────────────────────────
def fix_net_delta(df, col="net_out"):
    if df[col].mean() > 50000:
        print(f"    net_out looks cumulative (mean={df[col].mean():.0f}) — converting to delta KB/s")
        df[col] = df[col].diff().fillna(0).clip(lower=0) / 1024
    else:
        print(f"    net_out looks fine (mean={df[col].mean():.1f} KB/s) — no change needed")
    return df


# ── Helper: standardise columns ────────────────────────────────────────────────
def standardise(df, machine_id):
    df = df.rename(columns={
        "cpu_usage_percent":    "cpu_usage",
        "memory_usage_percent": "memory_usage",
        "disk_usage_percent":   "disk_io",
        "net_bytes_sent":       "net_out",
        "net_bytes_recv":       "net_in",
    })

    drop_cols = ["timestamp", "is_anomaly", "machine_id",
                 "scenario", "net_in", "net_bytes_recv", "process_spawn_rate_raw"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    for col in STANDARD_COLS:
        if col not in df.columns:
            print(f"    Column '{col}' missing — filling with 0")
            df[col] = 0

    df = df[STANDARD_COLS]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df[df < 0] = 0

    df["cpu_memory_ratio"] = df["cpu_usage"] / (df["memory_usage"] + 1e-5)
    df["machine_id"] = machine_id

    return df


# ── Helper: clean outliers from NORMAL data only ───────────────────────────────
def clean_normal(df, machine_id):
    before = len(df)
    df = df[df["cpu_usage"] < 96]
    removed = before - len(df)
    if removed > 0:
        print(f"    Removed {removed} rows with CPU>=96% from {machine_id} normal data")
    return df


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load and clean YOUR normal data
# ════════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Loading YOUR normal data...")
if not os.path.exists(MY_DATA):
    raise FileNotFoundError(f"Could not find your data at:\n  {MY_DATA}\nRun collect_my_data.py first.")

mine = pd.read_csv(MY_DATA)
print(f"    Loaded {len(mine)} rows")
mine = standardise(mine, "mine")
mine = fix_net_delta(mine, "net_out")
mine = clean_normal(mine, "mine")
print(f"    After cleaning: {len(mine)} rows")
print(f"    CPU avg: {mine['cpu_usage'].mean():.1f}%  "
      f"MEM avg: {mine['memory_usage'].mean():.1f}%  "
      f"NET avg: {mine['net_out'].mean():.1f} KB/s")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — Friend's normal data dropped
# ════════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Skipping friend's normal data (dropped — idle machine, CPU avg 0.3%)")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — Save combined (your data only)
# ════════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Saving combined dataset...")
combined = mine.copy()
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs(os.path.dirname(OUT_COMBINED), exist_ok=True)
combined.to_csv(OUT_COMBINED, index=False)

print(f"    Combined: {len(combined)} rows total")
print(f"      — yours : {len(mine)} rows")
print(f"    Saved → {OUT_COMBINED}")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — Process friend's anomaly data (for threshold calibration only)
# ════════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Processing friend's anomaly data...")
anomaly_count = 0
if not os.path.exists(FRIEND_ANOMALY):
    print(f"    [WARN] friend_anamoly.csv not found at:\n    {FRIEND_ANOMALY}")
    print("    Skipping — add the file and re-run this script.")
else:
    anomaly = pd.read_csv(FRIEND_ANOMALY)
    print(f"    Loaded {len(anomaly)} anomaly rows")
    anomaly = standardise(anomaly, "friend_anomaly")
    anomaly = fix_net_delta(anomaly, "net_out")
    # do NOT clean outliers — spikes are the point
    os.makedirs(os.path.dirname(OUT_ANOMALY), exist_ok=True)
    anomaly.to_csv(OUT_ANOMALY, index=False)
    anomaly_count = len(anomaly)
    print(f"    Saved {anomaly_count} anomaly rows → {OUT_ANOMALY}")
    print(f"    CPU avg: {anomaly['cpu_usage'].mean():.1f}%  "
          f"max: {anomaly['cpu_usage'].max():.1f}%")


# ════════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MERGE COMPLETE")
print("=" * 60)
print(f"  Normal training data : {len(combined)} rows → combined_behavior.csv")
print(f"  Anomaly calib data   : {anomaly_count} rows → anomaly_for_calibration.csv")
print("\n  Next step: run python src/feature_engineering.py")
print("=" * 60)