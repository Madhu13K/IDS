"""
collect_my_data.py
──────────────────
Collects YOUR machine's normal behavior. Appends to existing data.
process_spawn_rate = new processes per second (matches app.py formula).

Run from project root: python src/collect_my_data.py
"""

import psutil, pandas as pd, time, os
from datetime import datetime

TARGET_ROWS  = 600
INTERVAL_SEC = 2
SAVE_EVERY   = 50

OUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "data", "raw", "my_normal_behavior.csv"
)

# ── Delta tracking ─────────────────────────────────────────────────────────────
_last_net        = psutil.net_io_counters().bytes_sent
_last_net_t      = datetime.now()
_last_proc_count = len(psutil.pids())
_last_proc_t     = datetime.now()


def collect_snapshot():
    global _last_net, _last_net_t, _last_proc_count, _last_proc_t

    cpu  = psutil.cpu_percent(interval=1)
    mem  = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    now         = datetime.now()
    current_net = psutil.net_io_counters().bytes_sent
    elapsed_net = (now - _last_net_t).total_seconds() or 1
    net_out     = (current_net - _last_net) / 1024 / elapsed_net  # KB/s delta
    _last_net   = current_net
    _last_net_t = now

    current_proc      = len(psutil.pids())
    elapsed_proc      = (now - _last_proc_t).total_seconds() or 1
    proc_delta        = max(0, current_proc - _last_proc_count)
    spawn             = round(proc_delta / elapsed_proc, 3)        # processes/sec
    _last_proc_count  = current_proc
    _last_proc_t      = now

    return {
        "timestamp":          now.isoformat(),
        "cpu_usage":          round(cpu,     2),
        "memory_usage":       round(mem,     2),
        "disk_io":            round(disk,    2),
        "net_out":            round(net_out, 2),
        "process_count":      current_proc,
        "process_spawn_rate": spawn,
        "sudo_commands":      0,
        "failed_logins":      0,
        "is_anomaly":         0
    }


def print_progress(row_num, total, session_rows, row):
    bar_len = 28
    filled  = int(bar_len * row_num / total)
    bar     = "=" * filled + "-" * (bar_len - filled)
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


def save(existing_df, new_rows, path):
    new_df = pd.DataFrame(new_rows)
    final_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df
    final_df.to_csv(path, index=False)
    return final_df


def main():
    existing_rows = 0
    existing_df   = None

    if os.path.exists(OUT_FILE):
        existing_df   = pd.read_csv(OUT_FILE)
        existing_rows = len(existing_df)

    total_target = existing_rows + TARGET_ROWS

    print("=" * 70)
    print("  NORMAL BEHAVIOR COLLECTOR")
    print("=" * 70)
    print(f"  Existing rows : {existing_rows}")
    print(f"  This session  : {TARGET_ROWS} new rows (~{TARGET_ROWS*INTERVAL_SEC//60} minutes)")
    print(f"  Total after   : {total_target} rows")
    print(f"  Output        : {OUT_FILE}")
    print("\n  While this runs, use your machine normally:")
    print("  A) Browse + stream video")
    print("  B) Download a large file")
    print("  C) Compile code / run a heavy script")
    print("  D) Open multiple apps and use them\n")

    activity = input("  Which activity? (A/B/C/D or describe): ").strip()
    print(f"\n  Collecting during: {activity}")
    input("  Start your activity NOW, then press ENTER...\n")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    new_rows = []

    try:
        for i in range(1, TARGET_ROWS + 1):
            row = collect_snapshot()
            new_rows.append(row)
            print_progress(existing_rows + i, total_target, i, row)
            if i % SAVE_EVERY == 0:
                save(existing_df, new_rows, OUT_FILE)
            time.sleep(max(0, INTERVAL_SEC - 1))

    except KeyboardInterrupt:
        print(f"\n\n  Stopped early — {len(new_rows)} rows collected.")

    finally:
        if new_rows:
            final_df = save(existing_df, new_rows, OUT_FILE)
            new_df = pd.DataFrame(new_rows)
            print(f"\n  Session complete. Total rows: {len(final_df)}")
            print(f"  CPU avg: {new_df['cpu_usage'].mean():.1f}%  NET avg: {new_df['net_out'].mean():.1f} KB/s")
            print(f"\n  Next: python src/merge_datasets.py")


if __name__ == "__main__":
    main()
