"""
collect_my_data.py
──────────────────
Collects YOUR machine's normal behavior and appends to existing data.
Run this multiple times during different activities — it will never
overwrite what you already collected.

Target this session : 600 rows (~20 minutes at one row per 2 seconds)
Total target        : 2500+ rows across all sessions

Requirements:
    pip install psutil pandas

Run from project root:
    python src/collect_my_data.py
"""

import psutil
import pandas as pd
import time
import os
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_ROWS  = 600          # rows to collect THIS session (~20 minutes)
INTERVAL_SEC = 2            # seconds between snapshots
SAVE_EVERY   = 50           # auto-save every N rows

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "data", "raw", "my_normal_behavior.csv"
)

# ── Net delta tracking ─────────────────────────────────────────────────────────
_last_net   = psutil.net_io_counters().bytes_sent
_last_net_t = datetime.now()

def collect_snapshot():
    global _last_net, _last_net_t

    cpu  = psutil.cpu_percent(interval=1)
    mem  = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    proc = len(psutil.pids())

    now         = datetime.now()
    current_net = psutil.net_io_counters().bytes_sent
    elapsed     = (now - _last_net_t).total_seconds() or 1
    net         = (current_net - _last_net) / 1024 / elapsed  # KB/s
    _last_net   = current_net
    _last_net_t = now

    spawn = proc / max(cpu, 1)

    return {
        "timestamp":          now.isoformat(),
        "cpu_usage":          round(cpu,   2),
        "memory_usage":       round(mem,   2),
        "disk_io":            round(disk,  2),
        "net_out":            round(net,   2),
        "process_count":      proc,
        "process_spawn_rate": round(spawn, 3),
        "sudo_commands":      0,
        "failed_logins":      0,
        "is_anomaly":         0
    }

def print_progress(row_num, total, session_rows, row):
    bar_len = 28
    filled  = int(bar_len * row_num / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    pct     = row_num / total * 100
    remaining = (total - row_num) * INTERVAL_SEC
    print(
        f"\r  [{bar}] {pct:5.1f}%  "
        f"Session: {session_rows}  Total: {row_num}/{total}  |  "
        f"CPU {row['cpu_usage']:5.1f}%  "
        f"MEM {row['memory_usage']:5.1f}%  "
        f"NET {row['net_out']:6.1f} KB/s  |  "
        f"~{remaining//60:.0f}m {remaining%60:.0f}s left",
        end="", flush=True
    )

def main():
    # ── Load existing data if present ─────────────────────────────────────────
    existing_rows = 0
    existing_df   = None

    if os.path.exists(OUT_FILE):
        existing_df   = pd.read_csv(OUT_FILE)
        existing_rows = len(existing_df)

    total_target = existing_rows + TARGET_ROWS

    print("=" * 70)
    print("  NORMAL BEHAVIOR COLLECTOR — Append Mode")
    print("=" * 70)
    print(f"\n  Existing rows : {existing_rows}")
    print(f"  This session  : {TARGET_ROWS} new rows (~{TARGET_ROWS*INTERVAL_SEC//60} minutes)")
    print(f"  Total after   : {total_target} rows")
    print(f"  Output        : {OUT_FILE}")

    print("\n  ── DO THIS WHILE IT RUNS ────────────────────────────────────")
    print("  Pick ONE of these activities and do it the whole session:")
    print()
    print("  A)  Open 15+ Chrome tabs, stream YouTube fullscreen")
    print("  B)  Run a large file download (ubuntu ISO from ubuntu.com)")
    print("  C)  Compile code / run a heavy Python script in another terminal")
    print("  D)  Open VS Code + Chrome + Spotify all at once, use all three")
    print()
    print("  The goal is variety — heavier usage than your first collection.")
    print("  Press Ctrl+C at any time to stop and save.\n")
    print("  ─────────────────────────────────────────────────────────────\n")

    activity = input("  Which activity are you doing? (A/B/C/D or describe): ").strip()
    print(f"\n  Got it — collecting during: {activity}")
    input("  Start your activity NOW, then press ENTER to begin collecting...\n")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    new_rows = []

    try:
        for i in range(1, TARGET_ROWS + 1):
            row = collect_snapshot()
            new_rows.append(row)
            print_progress(existing_rows + i, total_target, i, row)

            # auto-save every SAVE_EVERY rows
            if i % SAVE_EVERY == 0:
                save(existing_df, new_rows, OUT_FILE)

            time.sleep(max(0, INTERVAL_SEC - 1))

    except KeyboardInterrupt:
        print(f"\n\n  Stopped early — collected {len(new_rows)} rows this session.")

    finally:
        if new_rows:
            final_df = save(existing_df, new_rows, OUT_FILE)
            print(f"\n  ── Session complete ──────────────────────────────────────")
            new_df = pd.DataFrame(new_rows)
            print(f"  This session:")
            print(f"    CPU  : avg {new_df['cpu_usage'].mean():.1f}%  "
                  f"min {new_df['cpu_usage'].min():.1f}%  "
                  f"max {new_df['cpu_usage'].max():.1f}%")
            print(f"    MEM  : avg {new_df['memory_usage'].mean():.1f}%")
            print(f"    NET  : avg {new_df['net_out'].mean():.1f} KB/s")
            print(f"  Total rows now: {len(final_df)}")
            print(f"\n  Run this script again during a DIFFERENT activity")
            print(f"  to keep adding variety. Aim for 2500+ rows total.")
            print(f"\n  When done collecting, run: python src/merge_datasets.py")
        else:
            print("\n  No rows collected — nothing saved.")

def save(existing_df, new_rows, path):
    new_df = pd.DataFrame(new_rows)
    if existing_df is not None:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df
    final_df.to_csv(path, index=False)
    return final_df

if __name__ == "__main__":
    main()