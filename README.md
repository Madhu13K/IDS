# IDS — Host-Based Intrusion Detection System

An unsupervised, machine-learning-based Intrusion Detection System that monitors real-time system behavior and flags anomalies using an Isolation Forest model trained on normal machine activity.

**Team:** K Madhushree · Akshaya Sreekumar · Ayisha Hemni Kalathingal  
**Repository:** [github.com/Madhu13K/IDS](https://github.com/Madhu13K/IDS)

---

## What It Does

The system continuously monitors 9 host-level features — CPU usage, memory usage, disk I/O, network output, process count, process spawn rate, sudo commands, failed login attempts, and a derived CPU/memory ratio. Each reading is scored by an Isolation Forest model. Readings that deviate significantly from learned normal behavior are flagged as intrusions.

A Flask API backend serves live metrics and manual injection endpoints. A browser-based dashboard visualises scores in real time, plots the last 40 readings against the detection threshold, and maintains a timestamped event log.

---

## Project Structure

```
IDS/
├── src/
│   ├── collect_my_data.py        # Step 1 — collect normal behavior data from your machine
│   ├── merge_datasets.py         # Step 2 — merge and standardise collected data
│   ├── feature_engineering.py    # Step 3 — feature engineering, scaling, save scaler
│   ├── train_model.py            # Step 4 — train Isolation Forest, calibrate threshold
│   ├── evaluate_model.py         # Step 5 — evaluate model (metrics, plots, report)
│   ├── app.py                    # Step 6 — Flask API (live /metrics and /inject endpoints)
│   ├── live_linux_detect.py      # Terminal-based live monitor (Linux/Mac)
│   ├── simulate_spikes.py        # Simulate attack scenarios against the trained model
│   ├── data_loader.py            # LEGACY — initial prototype, do not run
│   ├── validate_and_calibrate.py # LEGACY — superseded by train_model.py, do not run
│   ├── live_detect.py            # LEGACY — superseded by live_linux_detect.py
│   ├── check_threshold.py        # LEGACY — debug utility, do not run
│   └── test_env.py               # LEGACY — environment check, do not run
├── dashboard/
│   └── ids_dashboard.html        # Browser dashboard — open after starting app.py
├── data/
│   ├── raw/
│   │   ├── combined_behavior.csv         # merged normal training data
│   │   └── anomaly_for_calibration.csv   # second-machine data for threshold calibration
│   └── processed/
│       └── scaled_data.csv               # scaled features used for training
├── models/
│   ├── ids_model.pkl             # trained Isolation Forest model
│   ├── scaler.pkl                # fitted StandardScaler
│   └── threshold.txt             # calibrated decision threshold
├── reports/
│   ├── evaluation_report.txt     # precision, recall, F1, ROC-AUC
│   └── evaluation_plots.png      # confusion matrix, score distribution, ROC curve
└── README.md
```

---

## Requirements

Python 3.8 or higher. Install all dependencies:

```bash
pip install numpy pandas scikit-learn joblib flask flask-cors psutil matplotlib
```

For Windows Event Log support (sudo/failed login detection on Windows):

```bash
pip install pywin32
```

---

## How to Run — Step by Step

### Step 1 — Collect normal behavior data

Run this during typical machine usage (browsing, coding, downloading). Run multiple times during different activities to build variety. Target 2500+ rows total.

```bash
python src/collect_my_data.py
```

### Step 2 — Merge and standardise datasets

```bash
python src/merge_datasets.py
```

### Step 3 — Feature engineering and scaling

```bash
python src/feature_engineering.py
```

### Step 4 — Train the model

Trains Isolation Forest on your normal data and statistically calibrates the detection threshold.

```bash
python src/train_model.py
```

### Step 5 — Evaluate the model

Produces precision, recall, F1, ROC-AUC on a held-out test set. Saves report and plots to `reports/`.

```bash
python src/evaluate_model.py
```

### Step 6 — Start the Flask API

```bash
python src/app.py
```

### Step 7 — Open the dashboard

Open `dashboard/ids_dashboard.html` in your browser. The dashboard auto-polls Flask every 5 seconds and displays live anomaly scores.

---

## Simulate Attack Scenarios

With the model trained and Flask running, you can simulate known attack patterns:

```bash
python src/simulate_spikes.py
```

This scores predefined scenarios — CPU spike, brute-force login, cryptominer, reverse shell, process bomb — against the trained model and prints each verdict with its score.

You can also inject custom values directly from the dashboard using the **Simulate / Inject Sample** panel.

---

## Evaluation Results

Evaluated on a held-out test set (20% of normal data, never seen during training) combined with anomaly calibration data from a second machine.

| Metric | Value |
|---|---|
| Precision | 0.9756 |
| Recall | 0.9479 |
| F1 Score | 0.9615 |
| ROC-AUC | 0.9971 |
| False Positive Rate | 0.66% |
| Threshold | 0.02 |

Full report: `reports/evaluation_report.txt`  
Plots (confusion matrix, score distribution, ROC curve): `reports/evaluation_plots.png`

---

## How Detection Works

The model is trained exclusively on normal behavior using **Isolation Forest** — an unsupervised anomaly detection algorithm that isolates observations by randomly partitioning features. Anomalous points, being rare and different, are isolated in fewer partitions and receive lower scores.

Each live reading produces a **decision score**. Scores below the calibrated threshold are flagged as intrusions. The threshold is set at the midpoint between the 5th percentile of normal training scores and the 95th percentile of calibration scores, giving a statistically grounded boundary rather than a hardcoded value.

---

## Maintenance Process Model

This project follows the **Iterative Enhancement Model**. The system evolved through repeated cycles of data collection, feature engineering, model retraining, and threshold recalibration — each iteration improving on the previous. New features (`cpu_memory_ratio`, live auth.log parsing) were added incrementally without redesigning the pipeline. The feedback loop between collect → merge → engineer → train → evaluate → deploy directly maps to the iterative enhancement cycle.

---

## Known Limitations

- Training data is from a single Windows machine running VS Code and Chrome — the model reflects that specific workload and may not generalise to server environments or different operating systems
- The anomaly calibration dataset is behavioral data from a second machine with a distinct usage profile, not labelled attack traffic — Isolation Forest is unsupervised and detects deviation from learned normality, not specific attack signatures
- `sudo_commands` and `failed_logins` return 0 on Windows without `pywin32` installed, and require read access to `/var/log/auth.log` on Linux
- `process_spawn_rate` is an approximation computed as new PIDs per second between polling intervals
- The model requires retraining if deployed on a machine with significantly different normal behavior

---

## License

MIT
