# CMPS 364 — Detecting Eating Behavior from Wearable Sensors

Course project for **CMPS 364 — Advanced Machine Learning**.

This project studies eating behavior from wrist-worn inertial sensors using two complementary public datasets.

## Project Scope

We separate the project into two tasks:

```text
Clemson All-day     → eating-episode detection

KU Leuven FD-I      → bite/eating-gesture detection
                    → bite counting
                    → eating vs drinking confusion
                    → unseen-user generalization
                    → few-shot personalization
                    → CNN/RNN/Transformer comparison
                    → efficiency comparison
```

The two tasks are evaluated independently on their respective datasets.

We do **not** claim that the two datasets form a single jointly evaluated end-to-end system.

## Project Abstract

This project uses wrist-worn accelerometer and gyroscope data to study eating behavior with two complementary public datasets. KU Leuven FD-I is used to detect bites and sips, count eating gestures, compare neural architectures, measure eating speed, and test personalization for unseen users. Clemson All-day is used to detect complete eating episodes, estimate meals and eating time per day, and analyze where meal detection fails.

The gesture and meal detectors are trained and evaluated independently. The repository also includes a Sensor Logger converter and browser dashboard for replayed or live wrist-stream demonstrations.

## Key Results

| Result | Reported value |
|---|---:|
| KU Leuven unseen-test bite F1 | 0.817 (CNN-GRU) |
| KU Leuven unseen-test drink F1 | 0.837 (Dilated CNN) |
| Eating-predicted-as-drinking / drinking-predicted-as-eating | 0 / 0 |
| Fastest FD-I model | 5.7 ms per minute of data |
| Quantized CNN / causal-CNN size | 273 KB |
| Causal live-model look-ahead | 0 s |
| Clemson unseen-test episode F1 | 0.694 |
| Clemson unseen-test false alarms | 1.94 per day |
| Clemson meals under 5 min | 45.5% found |
| Clemson meals 30+ min | 85.7% found |

See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for the full comparison and [results_analysis.ipynb](results_analysis.ipynb) for the visual analysis.

```text
KU Leuven IMU -> 16 Hz preprocessing -> gesture model -> bites, sips, eating speed
Clemson IMU   -> 15 Hz preprocessing -> episode model  -> meals, meal starts/ends, daily totals
Phone stream  -> unit/axis conversion -> live gesture model -> browser dashboard
```

---

# Progress

Last updated: 2026-11-27.

Statuses: **done** (run on the real data), **code ready** (written and tested on synthetic data, no real-data results yet), **blocked**, **not started**. Run names refer to folders in `results/runs/`.

| Area | Item | Status | Result (test split unless noted) |
|---|---|---|---|
| Data | KU Leuven download, label checks, preparation, split, normalization | done | |
| Data | Clemson download, label checks, sensor alignment, split | done | |
| Data | Clemson model-ready preparation and normalization | done | 354 recordings, 93,546 windows, validation passed |
| Data | Check `P2118` / `P2154` (log about a day longer than sensor data) | done | Documented in alignment audits; no source data changed |
| Data | `config/fd1_folds.json` from the official `FD1_7_fold_id_list.zip` | done | 7 participant-independent folds |
| Q1 | Bite / sip / other per moment | done | CNN-GRU (`fd1_cnn_gru_cw_none`): drink F1 0.834, no eating/drinking mix-ups |
| Q2 | Bite count | done | CNN-GRU: bite F1 0.817, bite-count MAE 22.0 per participant |
| Q2 | Eating speed (bites per minute, time between bites) | done | CNN-GRU: 0.86 bites/min MAE, 1.80 s between-bite MAE |
| Q3 | Meal detection and start / end | done | `clemson_cnn_pw_none`: 118 of 165 meals found (71.5%), 1.94 false alarms/day, start / end error 4.1 / 6.5 min |
| Q4 | CNN vs CNN-GRU vs transformer | done | bite F1: CNN-GRU 0.817, CNN 0.776, causal CNN 0.751, transformer 0.623 |
| Q5 | New user, fine-tuning on 1 / 2 / 5 min of their data | code ready | Fold-0 results are in `results/runs/fd1_personalization_cnn_fold0/` |
| Q6 | Model size, CPU time and int8 | done | CNN and causal CNN 0.87 → 0.27 MB with no accuracy loss; CNN 5.7 ms vs CNN-GRU 103 ms of CPU per minute of data |
| Q7 | Meal-detector breakdown by meal type, context, BMI, eating hand | done | meals under 5 min found 45.5% vs 30+ min 85.7%; snacks 63.0% vs lunch 79.2% |
| Q9 | Meals per day and time spent eating | done | 3.30 vs 3.11 true meals/day (MAE 1.02); 57.6 vs 47.9 true eating min/day (MAE 25.9) |
| Q8 | Which hand is eating (nice to have) | not started | |
| Demo | Causal (no look-ahead) gesture model | done | `fd1_cnn_causal_cw_none`: bite F1 0.751, 0 s look-ahead |
| Demo | Trained checkpoints (`best.pt`) available to the demo machine | blocked | Checkpoints are intended for Google Drive, not Git |
| Demo | Phone-to-model converter, Sensor Logger on iOS and Android (live push, export zip, saved sessions) | code ready, checked with simulated pushes | |
| Demo | Live bite and meal output (replay and HTTP) and browser dashboard (`--ui`) | code ready, checked with replays | |
| Demo | Record a real phone session and check units / axes | blocked | Real-device phone/network validation is pending |
| Report | Results, privacy design, limitations | done | See the README sections below |

Real-data results: 18 training runs (13 KU Leuven, 5 Clemson), run on Colab and summarized in [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md). The notebook `results_analysis.ipynb` draws the comparison charts, which are saved in `results/figures/`. Tuning used validation only. Each final model was scored once on the test split.

Main finding: the original default, `--class-weight sqrt`, produced too many false bites on every KU Leuven model, and `--class-weight none` was better for all four (validation bite F1 for the causal CNN went from 0.433 to 0.734). `none` is now the default in `train_fd1.py` and `personalize_fd1.py`.

Treat the left-handed result in Q7 (4 of 12 meals found) as a hint only, because 12 meals is too few to conclude anything.

KU Leuven shows no left/right gap, so mirroring augmentation is not needed for bite detection:

- Training already has 1,310 left-wrist bites and 1,950 right-wrist bites, and 12 of the 24 training participants eat mostly with the left hand.
- On the test split, bite F1 by wrist is:
  - CNN-GRU (`fd1_cnn_gru_cw_none`): left 0.829, right 0.801
  - causal CNN (`fd1_cnn_causal_cw_none`): left 0.765, right 0.734

The questions behind Q1–Q9 are listed in [Project Goals and Plan](#project-goals-and-plan).

---

# Phase 1 — Dataset Validation and Reproducible Data Pipeline

The goal of Phase 1 was to make the data pipeline reproducible for all group members before model development.

Phase 1 includes:

1. Downloading both datasets from their official sources.
2. Verifying downloaded files and checksums.
3. Inspecting the actual released data formats.
4. Parsing and validating labels.
5. Aligning labels with sensor recordings.
6. Creating deterministic processed outputs.
7. Defining participant-independent train/validation/test splits.
8. Making sure another teammate can reproduce everything from a clean clone.

Phase 1 is complete. The `P2118` and `P2154` timestamp anomalies are retained as documented validation findings; no source data is silently corrected. See [Phase 1 Status](#phase-1-status).

---

# Repository Structure

```text
364-project/
│
├── config/
│   ├── datasets.json
│   ├── fd1_dev_split.json
│   ├── fd1_normalization.json
│   ├── clemson_dev_split.json
│   ├── clemson_normalization.json      (created by compute_clemson_normalization.py)
│   └── live_profiles.json              (sensor units per phone type and per model)
│
├── data/
│   ├── raw/                            (ignored by Git)
│   │   ├── clemson_allday/
│   │   └── kuleuven/
│   │
│   ├── processed/                      (ignored by Git)
│   │   ├── clemson_allday/
│   │   └── kuleuven_fd1/
│   │
│   └── metadata/
│
├── results/
│   └── data_audit/
│
├── scripts/
│   ├── data/
│   │   ├── download_data.py
│   │   │
│   │   ├── inspect_fd1_labels.py
│   │   ├── create_fd1_dev_split.py
│   │   ├── preprocess_fd1.py
│   │   ├── validate_fd1_processed.py
│   │   ├── compute_fd1_normalization.py
│   │   │
│   │   ├── inspect_clemson_allday.py
│   │   ├── inspect_clemson_event_markers.py
│   │   ├── inspect_clemson_sensors.py
│   │   ├── preprocess_clemson_events.py
│   │   ├── validate_clemson_alignment.py
│   │   ├── create_clemson_dev_split.py
│   │   ├── preprocess_clemson.py
│   │   ├── validate_clemson_processed.py
│   │   └── compute_clemson_normalization.py
│   │
│   ├── train/
│   │   ├── common.py
│   │   ├── models.py
│   │   ├── metrics.py
│   │   ├── postprocess.py
│   │   ├── train_fd1.py
│   │   ├── train_clemson.py
│   │   ├── compare_fd1_runs.py
│   │   ├── analyze_clemson_meals.py
│   │   ├── quantize_fd1.py
│   │   └── personalize_fd1.py
│   │
│   └── live/
│       ├── convert.py
│       ├── make_replay_from_fd1.py
│       ├── run_live.py
│       ├── ui_server.py
│       └── ui.html
│
├── .gitignore
├── README.md
└── requirements.txt
```

Model training and evaluation are organized under `scripts/train/` and `results/runs/`.

---

# Setup

## 1. Clone the repository

```powershell
git clone https://github.com/jesss77/364-project.git
cd 364-project
```

## 2. Create a virtual environment

From the repository root:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify Python:

```powershell
python --version
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Dependencies include:

- `numpy`
- `scipy`
- `openpyxl`
- `torch`
- `matplotlib`
- `ipykernel`

---

# Data Policy

Large dataset files are **not committed to GitHub**.

The repository contains:

- acquisition scripts,
- preprocessing scripts,
- configuration,
- metadata,
- checksums,
- validation summaries,
- documentation.

It does not contain the full raw datasets.

The following folders are intentionally ignored:

```text
data/raw/
data/processed/
.venv/
__pycache__/
```

Do not manually modify raw dataset files.

Any normalization, filtering, timestamp correction, label transformation, windowing, or exclusion must be implemented in tracked code.

## Model Checkpoints

Trained `best.pt` checkpoints are not committed to GitHub because they are large and can be redistributed outside the repository policy. Keep checkpoints in a shared Google Drive or another team-controlled location, and place them under the matching run folder when preparing a demo machine:

```text
results/runs/<run name>/config.json
results/runs/<run name>/metrics.json
results/runs/<run name>/best.pt
```

Live inference requires `best.pt` plus the run's `config.json` and normalization configuration. A run directory without `best.pt` can still be inspected for metrics but cannot load the model.

---

# Dataset 1 — KU Leuven FD-I

## Purpose

KU Leuven FD-I is used for:

- eating-gesture detection,
- bite counting,
- eating vs drinking discrimination,
- unseen-user evaluation,
- few-shot personalization,
- architecture comparison,
- efficiency experiments.

The dataset contains wrist IMU recordings from both hands.

---

## List available files

```powershell
python scripts\data\download_data.py --dataset kuleuven --list
```

Expected published files include:

```text
00_ReadMe.txt
FD-I.zip
FD-II.zip
FD1_7_fold_id_list.zip
MO.zip
```

For this project, the main required files are:

```text
FD-I.zip
FD1_7_fold_id_list.zip
00_ReadMe.txt
```

---

## Download the README

```powershell
python scripts\data\download_data.py --dataset kuleuven --file 00_ReadMe.txt
```

---

## Download FD-I

```powershell
python scripts\data\download_data.py --dataset kuleuven --file FD-I.zip
```

The downloader verifies the official MD5 checksum after download.

---

## Download the published participant folds

```powershell
python scripts\data\download_data.py --dataset kuleuven --file FD1_7_fold_id_list.zip
```

The official folds are used for participant-independent evaluation.

---

## Current validated KU Leuven results

The released FD-I label files have been validated against the dataset documentation.

Verified:

```text
Participants:       34
Total duration:     251.70 hours
Eating gestures:   4568
Drinking gestures: 1100
Sampling rate:      64 Hz
Labels:
    0 = other
    1 = eating
    2 = drinking
```

The left- and right-hand label arrays have matching participant lengths.

No unexpected label values were found.

---

## Validate FD-I labels

After the label files are available under:

```text
data/raw/kuleuven/FD-I/
```

run:

```powershell
python scripts\data\inspect_fd1_labels.py
```

The script generates:

```text
results/data_audit/fd1_summary.json
results/data_audit/fd1_participants.csv
```

These outputs confirm:

- participant count,
- recording duration,
- eating gesture count,
- drinking gesture count,
- left/right alignment,
- unexpected labels.

---

## Preprocess FD-I sensor data

```powershell
python scripts\data\create_fd1_dev_split.py
python scripts\data\preprocess_fd1.py
python scripts\data\validate_fd1_processed.py
python scripts\data\compute_fd1_normalization.py
```

`preprocess_fd1.py` extracts the `X_*.pkl` / `Y_*.pkl` files from `FD-I.zip` if they are not already extracted.

What the pipeline does:

```text
Split:          by participant, seed 42 → 24 train / 7 validation / 3 test
Wrists:         left and right kept as separate files (6 channels each, never merged)
Sample rate:    64 Hz → 16 Hz (scipy resample_poly; labels take every 4th point)
Windows:        60 s, 30 s stride (50% overlap) → 960 × 6 per window
Normalization:  per-channel mean/std from training participants only
```

Generated files:

```text
config/fd1_dev_split.json
config/fd1_normalization.json
data/processed/kuleuven_fd1/participants/{train,validation,test}/participant_XX_{L,R}.npz
data/processed/kuleuven_fd1/windows.csv
results/data_audit/fd1_preprocessing_summary.json
results/data_audit/fd1_processed_validation.json
```

Each `.npz` holds the full recording (`X`: float32 `[samples, 6]`, `y`: uint8 `[samples]`). `windows.csv` indexes the windows by start/end sample, so windows are not stored twice.

Validated:

```text
Participant/wrist files:  68 (34 × 2)
Windows:                  60,310 (train 42,788 / validation 12,100 / test 5,422)
Label share:              ~98.8% other, ~0.8–0.9% eating, ~0.3–0.35% drinking
NaN/Inf values:           none
```

The class imbalance is left as-is in the data and will be handled during training.

The 7 official FD-I folds (`FD1_7_fold_id_list.zip`) are stored in `config/fd1_folds.json` for Q5 cross-personalization. The separate development split above is used for architecture tuning and final held-out testing.

---

## KU Leuven limitation

FD-I contains gesture-level eating and drinking labels but does not release an independent meal start/end annotation file.

Therefore, KU Leuven is not used as the main source for eating-episode boundary evaluation.

That task is handled with Clemson All-day.

---

# Dataset 2 — Clemson All-day

## Purpose

Clemson All-day is used for:

- full-day eating-episode detection,
- meal/snack interval prediction,
- false-positive evaluation in everyday activity.

The transformed data contain six wrist-motion channels at 15 Hz.

---

## List required Clemson files

```powershell
python scripts\data\download_data.py --dataset clemson_allday --list
```

Expected output:

```text
ground_truth
transformed
demographics
```

These aliases correspond to:

```text
ground_truth → EVENTfiles.zip
transformed  → SHMfiles.zip
demographics → ShimmerDemographics.xlsx
```

---

## Download ground truth

```powershell
python scripts\data\download_data.py --dataset clemson_allday --file ground_truth
```

This downloads:

```text
data/raw/clemson_allday/EVENTfiles.zip
```

---

## Download demographics

```powershell
python scripts\data\download_data.py --dataset clemson_allday --file demographics
```

This downloads:

```text
data/raw/clemson_allday/ShimmerDemographics.xlsx
```

---

## Download transformed sensor data

```powershell
python scripts\data\download_data.py --dataset clemson_allday --file transformed
```

This downloads approximately:

```text
4.56 GB
```

to:

```text
data/raw/clemson_allday/SHMfiles.zip
```

The Clemson downloader records a SHA-256 checksum in the metadata manifest.

---

# Clemson Ground-Truth Validation

The public ground-truth archive has already been validated.

Verified:

```text
Recording files:       354
Eating-event rows:     1133
Eating-event duration: 265.24 hours
```

Validation results:

```text
Missing START markers:    0
Missing END markers:      0
Malformed markers:        0
Malformed events:         0
Nonpositive intervals:    0
Out-of-bounds intervals:  0
Overlapping events:       0
```

One recording ID, `P2484`, is not present in the demographics workbook.

It is retained rather than silently removed.

---

# Clemson Timestamp Handling

Some Clemson files use timestamps such as:

```text
2015-10-29 24:12:41
```

This means:

```text
2015-10-30 00:12:41
```

The preprocessing code handles this automatically.

Raw source files are never modified.

---

# Inspect Clemson Ground Truth

To inspect the structure of the released event archive:

```powershell
python scripts\data\inspect_clemson_allday.py
```

This checks:

- file count,
- participant/recording IDs,
- event-file structure,
- demographics structure,
- basic metadata consistency.

---

# Marker Audit

A marker-specific inspection script is also available:

```powershell
python scripts\data\inspect_clemson_event_markers.py
```

This was used to verify inconsistent whitespace in Clemson `START` and `END` lines.

It is mainly a validation/debugging utility.

---

# Preprocess Clemson Events

Run:

```powershell
python scripts\data\preprocess_clemson_events.py
```

This converts the raw text event files into normalized tables.

Generated files:

```text
data/processed/clemson_allday/events.csv
data/processed/clemson_allday/recordings.csv
```

Validation results are saved to:

```text
results/data_audit/clemson_events_validation.json
```

---

# Clemson Processed Files

## `events.csv`

Contains one row per eating event.

Important fields include:

```text
recording_id
event_number
event_type
event_start
event_end
start_offset_seconds
end_offset_seconds
duration_seconds
location
seconds_status
social_context
food
activity
```

The offset fields represent time relative to the start of the corresponding recording.

These offsets are mapped to sensor samples in the Clemson preprocessing pipeline.

At 15 Hz:

```text
sample_index = round(time_offset_seconds × 15)
```

This mapping has been checked against the transformed sensor files (see below).

---

## `recordings.csv`

Contains one row per all-day recording.

Important fields include:

```text
recording_id
recording_start
recording_end
recording_duration_seconds
event_count
```

---

# Clemson Sensor Validation and Alignment

```powershell
python scripts\data\inspect_clemson_sensors.py
python scripts\data\validate_clemson_alignment.py
```

`inspect_clemson_sensors.py` checks the `.shm` file layout. `validate_clemson_alignment.py` matches every eating event to its sensor recording.

Verified:

```text
Sensor files:        354 (one per recording, no missing or extra IDs)
Layout:              little-endian float32, 6 channels per sample, 15 Hz
Channels:            linear_acc_x, linear_acc_y, linear_acc_z, gyro_yaw, gyro_pitch, gyro_roll
Total sensor time:   4,704 hours
Events inside data:  1,132 of 1,133
```

Known issues, kept in the audit outputs rather than hidden:

- `P2055` event 2 (Lunch) ends 0.27 s after the sensor data ends. Preprocessing clips it to the sensor length.
- In 117 recordings the sensor data is shorter than the START/END log. The median difference is 0.2 s.
- In a few recordings the log is about a day longer than the sensor data. For example, `P2154` differs by 86,400.7 s and `P2118` by 94,739 s. This looks like a date error in the log. All events in these recordings still fall inside the sensor data, but they should be checked before the results are reported.

Generated files:

```text
results/data_audit/clemson_sensor_archive.json
results/data_audit/clemson_alignment_summary.json
results/data_audit/clemson_recording_alignment.csv
results/data_audit/clemson_event_alignment.csv
```

---

# Clemson Model-Ready Preprocessing

```powershell
python scripts\data\create_clemson_dev_split.py
python scripts\data\preprocess_clemson.py
python scripts\data\validate_clemson_processed.py
python scripts\data\compute_clemson_normalization.py
```

`create_clemson_dev_split.py` only needs the tracked `clemson_recording_alignment.csv`. Its output, `config/clemson_dev_split.json`, is already committed.

The other three scripts need `SHMfiles.zip` and `events.csv`.

What the pipeline does:

```text
Split:          by recording ID, seed 42 → 248 train / 53 validation / 53 test (70/15/15)
Sample rate:    15 Hz, not resampled
Labels:         per sample, 1 = inside a logged eating event, 0 = otherwise
                (overlapping events are merged; events are clipped to the sensor length)
Windows:        6 min (5,400 samples), 3 min stride (50% overlap)
Normalization:  per-channel mean/std from training recordings only
```

Each Clemson recording ID has one recording, and the demographics workbook has no other participant key. The split therefore assumes one recording ID per person. If the same person was recorded under two IDs, that person could appear in more than one split.

Recordings shorter than 6 minutes produce no windows. They are listed in the preprocessing summary.

Generated files:

```text
config/clemson_dev_split.json
config/clemson_normalization.json
data/processed/clemson_allday/recordings/{train,validation,test}/PXXXX.npz
data/processed/clemson_allday/windows.csv
results/data_audit/clemson_preprocessing_summary.json
results/data_audit/clemson_processed_validation.json
```

`validate_clemson_processed.py` checks:

- the file count and split folders,
- shapes and dtypes,
- the sample count against the alignment audit,
- for NaN/Inf values,
- that the labels match `events.csv`,
- window length, stride and per-recording window count,
- the eating counts in `windows.csv`.

The full processed output is about 6 GB.

---

# Metadata and Checksums

Download metadata is stored under:

```text
data/metadata/
```

Examples include:

```text
kuleuven_dataverse.json
clemson_allday_files.json
clemson_allday_downloads.json
```

These files are useful for:

- recording official URLs,
- confirming dataset versions,
- recording published checksums,
- recording locally generated SHA-256 hashes,
- verifying reproducibility across group members.

---

# Validation Outputs

Lightweight validation results are stored under:

```text
results/data_audit/
```

Examples:

```text
fd1_summary.json
fd1_participants.csv
fd1_preprocessing_summary.json
fd1_processed_validation.json
clemson_allday_structure.json
clemson_event_marker_audit.json
clemson_events_validation.json
clemson_sensor_archive.json
clemson_alignment_summary.json
clemson_recording_alignment.csv
clemson_event_alignment.csv
clemson_preprocessing_summary.json
clemson_processed_validation.json
```

These files are small enough to track in Git and provide evidence that the dataset assumptions were checked against the actual released files.

---

# Reproducing Phase 1 from a Clean Clone

A teammate starting from scratch should follow this order.

## Step 1 — Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Step 2 — Inspect dataset sources

```powershell
python scripts\data\download_data.py --dataset kuleuven --list
python scripts\data\download_data.py --dataset clemson_allday --list
```

## Step 3 — Download KU Leuven files

```powershell
python scripts\data\download_data.py --dataset kuleuven --file 00_ReadMe.txt

python scripts\data\download_data.py --dataset kuleuven --file FD-I.zip

python scripts\data\download_data.py --dataset kuleuven --file FD1_7_fold_id_list.zip
```

## Step 4 — Download Clemson files

```powershell
python scripts\data\download_data.py --dataset clemson_allday --file ground_truth

python scripts\data\download_data.py --dataset clemson_allday --file demographics

python scripts\data\download_data.py --dataset clemson_allday --file transformed
```

## Step 5 — Validate Clemson labels

```powershell
python scripts\data\inspect_clemson_allday.py
python scripts\data\preprocess_clemson_events.py
```

## Step 6 — Validate Clemson sensors and alignment

```powershell
python scripts\data\inspect_clemson_sensors.py
python scripts\data\validate_clemson_alignment.py
```

## Step 7 — Build model-ready Clemson data

```powershell
python scripts\data\create_clemson_dev_split.py
python scripts\data\preprocess_clemson.py
python scripts\data\validate_clemson_processed.py
python scripts\data\compute_clemson_normalization.py
```

## Step 8 — Validate KU Leuven labels

After extracting the required FD-I label files:

```powershell
python scripts\data\inspect_fd1_labels.py
```

## Step 9 — Build model-ready KU Leuven data

```powershell
python scripts\data\create_fd1_dev_split.py
python scripts\data\preprocess_fd1.py
python scripts\data\validate_fd1_processed.py
python scripts\data\compute_fd1_normalization.py
```

Both validation scripts stop with an error and list every problem they find. A clean run ends with `VALIDATION PASSED`.

---

# Phase 1 Status

## Completed

- Dataset selection
- Dataset registry
- Reproducible KU Leuven acquisition
- Reproducible Clemson acquisition
- KU Leuven label validation
- KU Leuven published fold validation
- KU Leuven sensor extraction, 16 Hz resampling and 60 s windowing
- KU Leuven processed-data validation
- KU Leuven development split (24 / 7 / 3) and training-only normalization
- Clemson event archive inspection
- Clemson demographics inspection
- Clemson timestamp handling
- Clemson eating-event preprocessing
- Clemson ground-truth validation
- Clemson sensor format inspection (float32, 6 channels, 15 Hz)
- Clemson event/sensor alignment (1,132 of 1,133 events inside sensor data)
- Clemson development split (248 / 53 / 53)
- Clemson model-ready preprocessing, validation and normalization, run on the full `SHMfiles.zip` (354 recordings, 93,546 windows, validation passed)

## Remaining

- Real-phone calibration of sensor units and wrist-axis orientation
- Full seven-fold result aggregation for Q5

---

# Important Reproducibility Rules

1. Do not manually edit raw dataset files.
2. Do not commit raw datasets to GitHub.
3. Do not commit generated processed datasets.
4. Keep participant splits independent.
5. Split participants before fitting scalers or creating overlapping training windows.
6. Fit normalization statistics using training data only.
7. Do not use test participants for threshold tuning.
8. Any exclusion or correction must be implemented in code and documented.
9. Dataset quirks must be preserved in validation outputs rather than silently hidden.
10. A clean clone should be able to recreate all processed outputs using tracked scripts.

---

# Phase 2 — Baseline Models

Training code lives in `scripts/train/`:

```text
common.py                 paths, seeding, device choice, window dataset, full-recording inference
models.py                 KU Leuven gesture models (cnn, cnn_gru, transformer; cnn can be causal) and EpisodeCNN (Clemson)
metrics.py                confusion matrix, precision/recall/F1, segment extraction and matching
postprocess.py            bites from labels, eating speed, meals from window probabilities (shared with the live demo)
train_fd1.py              KU Leuven gesture model (Q1, Q2; --model for Q4, --causal for the demo)
train_clemson.py          Clemson meal model (Q3, Q9)
compare_fd1_runs.py       table of all KU Leuven runs → results/fd1_model_comparison.csv (Q4)
analyze_clemson_meals.py  meal detection by meal type, context and demographics (Q7)
quantize_fd1.py           int8 conversion: size, CPU time, accuracy before and after (Q6)
personalize_fd1.py        new-user experiment: fine-tune on 1 / 2 / 5 min of a held-out person (Q5)
```

Live demo code lives in `scripts/live/` (see [Live demo](#live-demo)).

Install PyTorch first (`pip install -r requirements.txt` now includes `torch`). Both scripts pick CUDA, then Apple MPS, then CPU unless `--device` is given.

The first time a recording is read, its `.npz` file is copied to `data/processed/<dataset>/cache/` as `.npy` files. Later reads use memory mapping, so the full Clemson data never has to fit in RAM.

Each run writes the following to `results/runs/<run name>/`:

- `config.json`
- `history.json`
- `metrics.json`
- `best.pt` (not committed)

## KU Leuven gesture CNN

```powershell
python scripts\data\compute_fd1_normalization.py
python scripts\train\train_fd1.py
```

- Input: 60 s windows (960 × 6). Output: a label (other / eating / drinking) for every sample.
- Model: a residual stack of dilated 1D convolutions with a receptive field of 253 samples (about 16 s) and about 209k parameters.
- Imbalance: `--class-weight none` (plain cross-entropy) is the default. `sqrt` and `inverse` weight eating and drinking up by the square root of, or the full, inverse class frequency. On the real data `sqrt` gave too many false bites on every model (see [Progress](#progress)).
- Model selection: the mean of eating F1 and drinking F1, measured per sample on full validation recordings.
- Scoring always runs on full recordings, not windows, so overlapping windows are not counted twice.
  - Per-sample confusion matrix and F1 for each class.
  - Event-level F1 for bites and drinks. Predicted segments shorter than 0.5 s are dropped, and segments less than 0.25 s apart are merged. Each true event can be matched at most once.
  - Eating events predicted as drinking, and drinking events predicted as eating.
  - Bite-count error per participant (both wrists summed) and per wrist file: MAE and mean relative error.

## KU Leuven architecture comparison

All three models take the same input, give the same per-sample output, and are trained and scored the same way. Only the network in the middle changes.

```powershell
python scripts\train\train_fd1.py --model cnn --run-name fd1_cnn
python scripts\train\train_fd1.py --model cnn_gru --run-name fd1_cnn_gru
python scripts\train\train_fd1.py --model transformer --run-name fd1_transformer
python scripts\train\compare_fd1_runs.py
```

| `--model` | What it is | Parameters | Context used at inference |
|---|---|---|---|
| `cnn` | 5 residual blocks of dilated 1D convolutions | 209k | 253 samples (receptive field) |
| `cnn_gru` | conv stem + 2 dilated blocks, then a bidirectional GRU (64 units per direction) | 135k | 15 s on each side |
| `transformer` | conv stem, 4× downsampling to 4 Hz tokens, 3 pre-norm encoder layers (4 heads, sinusoidal positions), upsampling back to 16 Hz with a skip connection | 156k | 15 s on each side |

- The GRU and the transformer score full recordings in 30 s steps, with 15 s of context on each side, so every forward pass sees 60 s, the same length they were trained on. The CNN runs on 15-minute chunks.
- `metrics.json` also records the model size (float32 MB) and the single-thread CPU time to label one minute of data. This is a first, rough efficiency number before the int8 work.
- On Apple MPS the GRU trains slower than on CPU. Add `--device cpu` for `cnn_gru` on a Mac. On a CUDA GPU (for example Colab) it runs at normal speed.
- The bidirectional GRU and the transformer look at future samples. That is fine for offline scoring, but an on-watch version would need a causal variant or a short delay.
- `compare_fd1_runs.py` reads every KU Leuven run in `results/runs/` (or only the run names you pass) and writes one row per run with validation and, if present, test scores.

## Clemson episode CNN

```powershell
python scripts\data\compute_clemson_normalization.py
python scripts\train\train_clemson.py
```

- Input: 6 min windows (5,400 × 6). Target: the fraction of the window that is eating (soft BCE target, `pos_weight` = √(non-eating / eating)).
- Model: four convolution blocks with pooling, then global average pooling and one output (about 58k parameters).
- Episode reconstruction:
  1. Window probabilities are averaged onto a 3-minute timeline.
  2. A hysteresis rule is applied: an episode starts when the probability reaches the start threshold and continues while it stays at or above the end threshold.
  3. Episodes less than 3 minutes apart are merged, and episodes shorter than 3 minutes are dropped.
- The start and end thresholds are tuned on validation (best episode F1, with ties broken by time F1) and then applied unchanged to test.
- Scoring:
  - Episodes found (true positive rate) and false alarms per hour and per day.
  - Episode F1.
  - Mean start and end error of matched episodes.
  - Per-sample time F1.

## Eating speed (Q2)

`train_fd1.py` also reports eating speed for each participant, from predicted bites and from true bites:

- Bites less than 60 s apart (`--bout-gap-seconds`) form one eating bout. Bouts with fewer than 3 bites are ignored.
- Bites per minute = bites in bouts / total bout time.
- Time between bites = the median gap between bite starts inside bouts.
- `metrics.json` gives the MAE of both between predicted and true, over participants (both wrists combined).

The final run also writes `predictions_<split>.csv`: every predicted and true bite and sip, with start and end times and whether it was matched.

## Clemson meal outputs, daily totals (Q9) and breakdowns (Q7)

The final Clemson run writes `meals_<split>.csv`: every predicted and true meal, with start and end times and its matched partner. `metrics.json` also has `daily_totals`: mean meals per day and eating minutes per day, predicted vs true, and their MAE. Each Clemson recording is one day.

```powershell
python scripts\train\analyze_clemson_meals.py <run name> --split validation
```

This joins `meals_<split>.csv` with `events.csv` and `ShimmerDemographics.xlsx` and reports, per group, how many logged meals were found and the mean start and end error. The groups are:

- meal type (breakfast / lunch / dinner / snack / other)
- meal length
- location
- social context
- activity
- time precision (logged with or without seconds)
- BMI, age, gender and eating hand

False alarms per day are also broken down by BMI, age, gender and eating hand. Groups with fewer than 10 meals are merged into "other (small groups)" (`--min-group`). Output: `meal_breakdown_<split>.json` in the run folder.

## Causal gesture model (for the live demo)

```powershell
python scripts\train\train_fd1.py --model cnn --causal --class-weight none --run-name fd1_cnn_causal_cw_none
```

`--causal` pads every convolution on the left only, so each label uses only past readings and the live demo can label the newest reading immediately. Each run records its look-ahead in `config.json` and `metrics.json`:

| Model | Look-ahead |
|---|---|
| causal CNN | 0 s |
| CNN | 7.9 s |
| CNN-GRU and transformer | 15 s |

## int8 (Q6)

```powershell
python scripts\train\quantize_fd1.py <run name>
```

- **What it does:** converts a trained KU Leuven model to int8 and scores both versions on validation (add `--eval-test` for the final model).
  - CNN models use static int8, calibrated on 256 training windows.
  - The transformer falls back to dynamic int8 on its linear layers.
- **What it records:** file size, single-thread CPU time per minute of data, eating F1, bite F1 and bite-count error, in `quantization.json`.
- **Synthetic-data check:** the CNN goes from 0.87 MB to 0.27 MB with the same scores.
- **Speed:** the int8 engine on desktop CPUs (`qnnpack`) is not faster than float32, so int8 speed has to be judged on a phone or watch chip.
- **Version warning:** PyTorch marks `torch.ao.quantization` as deprecated. It works in torch 2.15.

## New users (Q5)

```powershell
python scripts\train\personalize_fd1.py --model cnn
```

1. For each fold, train a base model on all other participants.
2. For each held-out participant, start 30 s before their first bite and take 1, 2 or 5 minutes of their labeled data (both wrists).
3. Fine-tune on it in two ways:
   - only the last layer (`head`)
   - all layers (`all`), with BatchNorm statistics frozen
4. Score the base model and every fine-tuned version on the same part of the recording, after the longest budget.

Output: `personalization.json` with per-participant scores and averages.

The official folds come from `config/fd1_folds.json` (`{"source": ..., "folds": [[ids], ...]}`), generated from `FD1_7_fold_id_list.zip`. `--random-folds N` makes seeded folds for testing only.

Example fold-zero run:

```powershell
python scripts\train\personalize_fd1.py --model cnn --class-weight none --run-name fd1_personalization_cnn_fold0 --folds 0
```

The completed fold-zero result is in `results/runs/fd1_personalization_cnn_fold0/personalization.json`.

## Live demo

```text
phone on wrist ──► convert.py ──► 16 Hz, KU Leuven units ──► gesture model ──► bites and sips (near-live)
                         └──────► 15 Hz, Clemson units ───► meal model ────► meal started / ended (every 3 min)
```

### Phone setup with Sensor Logger (iOS and Android)

The same setup works on both platforms. It follows Sensor Logger's own documentation ([PUSHING.md](https://github.com/tszheichoi/awesome-sensor-logger/blob/main/PUSHING.md), [UNITS.md](https://github.com/tszheichoi/awesome-sensor-logger/blob/main/UNITS.md)):

- `accelerometer` is acceleration without gravity, in m/s².
- `gravity` is a separate vector, in m/s².
- `gyroscope` is in rad/s.
- The default profile, `sensor_logger`, adds gravity back for the KU Leuven model and leaves it out for the Clemson model.

1. Put the laptop and the phone on the same Wi-Fi. A managed work laptop may block incoming connections even when its firewall allows Python. Use a personal laptop if so.
2. On the laptop, run:

   ```powershell
   python scripts\live\run_live.py --gesture-run fd1_cnn_causal_cw_none --listen 8000 --ui 8050 --save session.jsonl
   ```

   It prints the push URL to use, for example `http://192.168.1.20:8000/`. Open `http://localhost:8050/` for the dashboard.
3. Install Sensor Logger (App Store or Google Play).
4. On the Logger page, turn on **Accelerometer**, **Gravity** and **Gyroscope**. Everything else can stay off.
5. Open settings (the gear icon on the Logger page):
   - Turn on **HTTP Push** and enter the push URL.
   - Turn on **Standardise Units & Frames**, so iOS and Android send the same units and axes.
   - Tap **Tap to Test Pushing**. The laptop should print `Test message received`.
6. Start recording. The laptop prints `First data from the phone...`, and the dashboard starts moving.
7. **Check:** lay the phone flat, screen up, on a table. In the dashboard (or with `--show-stream 5`), the accelerometer should read about (0, 0, +9.8) m/s².
   - About −9.8 on z: Standardise Units & Frames is probably off.
   - Large values on x or y: the phone's axes don't match. Fix it with `--axes`.
8. For the demo, wear the phone on the eating wrist, screen facing out on the back of the wrist.

How stopping works:

- The first Ctrl-C ends the stream and prints the summary.
- With `--ui`, the dashboard stays open until a second Ctrl-C.
- `--save` keeps every pushed message, so the session can be replayed exactly with `--replay session.jsonl`.

Without a network, record in Sensor Logger and export it as zipped CSV, then replay the export:

```powershell
python scripts\live\run_live.py --gesture-run fd1_cnn_causal_cw_none --replay recording.zip --speed 1 --ui 8050
```

The export needs `Accelerometer.csv`, `Gyroscope.csv` and `Gravity.csv`, in a zip or an unzipped folder.

### Other inputs and checks

`config/live_profiles.json` also has the following profiles, for other apps:

| Profile | Accelerometer | Gyroscope | Gravity |
|---|---|---|---|
| `android` | m/s² | rad/s | included |
| `iphone_total` | g | rad/s | included |
| `iphone_user` | g | rad/s | separate |

The model targets (`fd1`, `clemson`) are marked `"verified": false`, because their units were inferred from the data. The axis directions of the original wrist sensors are also unknown, so a phone may need `--axes` to match them. Checking this needs a trained model and a short recording of real eating.

`convert.py` does the following:

- converts units
- remaps axes (`--axes`, for example `-y,x,z`)
- removes gravity with a 1 s low-pass estimate when the target needs it, or adds a separate gravity vector
- averages the readings into 16 or 15 per second

`make_replay_from_fd1.py` turns a KU Leuven recording into a phone-style CSV (50 readings per second) with its true labels. `run_live.py` prints the true counts at the end of such a replay. `--listen` also accepts a simple format, `{"time_s": [...], "acc": [[x, y, z], ...], "gyro": [...], "gravity": [...]}`.

Synthetic-data checks:

- **Same result on all three input paths:** a 15-minute recording with 22 true bites and 6 sips gave 21 bites and 6 sips through each of these:
  - a CSV replay
  - simulated Sensor Logger HTTP pushes (4,500 batched messages with accelerometer, gravity, gyroscope and orientation entries, one delivered out of order)
  - a simulated Sensor Logger export zip
- **Wrong profile:** reading the CSV with the wrong unit profile gave 54 bites and 67 sips.
- **Meals:** a replayed 4-hour fake Clemson day reported both meals within 1 minute of their true times.

## Live dashboard

```powershell
python scripts\live\run_live.py --gesture-run fd1_cnn_causal_cw_none --meal-run clemson_cnn_pw_none --replay replay\p6L.csv --speed 1 --ui 8050
```

Then open `http://localhost:8050/` in a browser on the same computer. The dashboard is served only on `localhost`, so it needs no firewall changes. It works with `--replay` and with `--listen`, and it polls the runner 4 times a second. It shows:

- **Tiles:**
  - bites and sips, with the time of the last one
  - eating speed (bites per minute)
  - median time between bites
  - meal status: eating since a time, not eating, or off when no meal model is loaded
- **Signal charts:** accelerometer (m/s²) and gyroscope (°/s) after conversion, for the last 30 s.
- **Model chart:** eating and drinking scores for each time point, with shaded blocks for detected bites and sips and a label strip underneath.
- **Event log:** newest first, including meal started and meal ended lines.
- **Meal score chart:** one bar per 3-minute step, with the start and end thresholds as a dashed line.
- **Footer:** phone readings per second, model readings per second, and the model's look-ahead.

When a replay ends, the page shows "Finished" and stays open until Ctrl-C.

`train_fd1.py` and `train_clemson.py` now save the scaling numbers they used in each run's `config.json`. `run_live.py` uses those numbers when they are present, so a model always sees data scaled the way it was trained. Before this, a model trained on synthetic data found no bites when run from the repo.

Live streaming from a phone to a managed work laptop can be blocked by its security software, even with the macOS firewall allowing Python. Use a personal laptop for the phone demo, or record on the phone and replay the file.

## Privacy and Ethics

- Raw recordings and generated processed datasets are excluded from Git and should remain under the user's local control or an approved team storage location.
- Dataset participants are identified only by dataset-assigned IDs. The models do not produce or store names or other direct identities.
- Demographics are used only for aggregate failure analysis. They must not be used to identify or profile individual people.
- Age, gender, BMI, eating hand, location, and social context should be reported only in anonymized aggregate summaries.
- Live sensor streams and JSONL recordings can reveal detailed movement and timing. Obtain consent before recording a participant and delete demo recordings after use.
- The system is a research prototype, not a clinical eating, nutrition, or health-monitoring device. It does not diagnose eating disorders or provide medical advice.
- Do not commit live recordings, credentials, private URLs, or participant identifiers to this repository.

## Limitations

- Clemson detection drops for meals under five minutes: only 45.5% are found, compared with 85.7% for meals lasting 30 minutes or longer.
- The Clemson left-handed result is based on only 12 meals and should be treated as an exploratory hint rather than a stable estimate.
- Clemson age, BMI, context, and eating-hand subgroups can contain few recordings; small-group percentages should not be generalized broadly.
- The bidirectional GRU and transformer require future context in their offline form, while the causal CNN is the model suitable for immediate live display.
- Phone unit, gravity, and axis mappings still require real eating recordings for full validation. The original sensor axes and phone-axis correspondence are not assumed to be identical.
- Int8 comparisons use desktop CPU timing. Watch or phone-chip performance has not yet been measured.
- KU Leuven and Clemson use different units, rates, data formats, and labels, so their outputs are not fused or jointly trained.
- The Clemson meal model is not validated across every environment, meal style, device placement, or cultural eating pattern.

## Test split

Neither script scores the test split unless `--eval-test` is given. Use it only for the final run of a model that will be reported. All tuning uses validation only.

## Quick check

For a short run, for example on a laptop, add `--epochs 2 --max-train-windows 2000`.

---

# Reproducing the Phase 2 Results

After downloading and preprocessing both datasets:

```powershell
python scripts\train\train_fd1.py --model cnn_gru --class-weight none --run-name fd1_cnn_gru_cw_none
python scripts\train\train_fd1.py --model cnn --class-weight none --run-name fd1_cnn_cw_none
python scripts\train\train_fd1.py --model cnn --causal --class-weight none --run-name fd1_cnn_causal_cw_none
python scripts\train\train_fd1.py --model transformer --class-weight none --run-name fd1_transformer_cw_none

python scripts\train\train_clemson.py --pos-weight none --run-name clemson_cnn_pw_none
python scripts\train\compare_fd1_runs.py
python scripts\train\quantize_fd1.py fd1_cnn_gru_cw_none --eval-test
python scripts\train\analyze_clemson_meals.py clemson_cnn_pw_none --split test
```

Run test evaluation only after model and threshold choices are fixed on validation data.

For the live demo replay:

```powershell
python scripts\live\run_live.py --gesture-run fd1_cnn_causal_cw_none --replay replay.jsonl --ui 8050
```

For both live gesture and meal output:

```powershell
python scripts\live\run_live.py --gesture-run fd1_cnn_causal_cw_none --meal-run clemson_cnn_pw_none --listen 8000 --ui 8050 --save session.jsonl
```

# Project Goals and Plan

## End goal

1. Models trained to answer the questions below, scored on the real datasets.
2. A converter that turns the live sensor stream of a phone worn on the wrist into the training format: same channels, units, axis directions, readings per second, gravity handling and scaling.
3. A live or near-live output while someone eats:

```text
12:31:05  meal started
12:31:40  bite 1
12:31:52  bite 2
12:32:10  sip
...
12:52:30  meal ended — 38 bites, 3 sips, 21 min, 1.8 bites/min
```

Bites and sips can be shown within a few seconds (KU Leuven model). Meal start and end lag by 3–6 minutes, because the Clemson model works on 6-minute windows.

## Questions

Status of each question is tracked in [Progress](#progress).

| # | Question | Dataset | Scored with | In the live demo |
|---|---|---|---|---|
| Q1 | Is this movement a bite, a sip, or something else? | KU Leuven | per-label F1, eating vs drinking confusion | yes |
| Q2 | How many bites, and how fast? | KU Leuven | bite F1, bite-count error per person, bites/min and time-between-bites error | yes |
| Q3 | Is the person eating now, and when did each meal start and end? | Clemson | meals found, false alarms per day, start/end error | semi-live |
| Q4 | Which model works best: CNN, GRU or transformer? | KU Leuven | Q1/Q2 scores against size and speed | picks the demo model |
| Q5 | Does it work for someone new, and how much of their data fixes it? | KU Leuven, 7 official folds | Q1/Q2 before and after 1, 2 or 5 min of their data | fine-tune on the demo person |
| Q6 | Can it run on a watch? | KU Leuven | model size, CPU time per minute, accuracy after int8 | indirectly |
| Q7 | Where does the meal detector fail, and for whom? | Clemson | detection and false alarms by meal type, context, BMI, eating hand | no |
| Q9 | How many meals per day, and how much time spent eating? | Clemson | daily totals vs logged meals | yes |

Nice to have, only if time allows:

| # | Question | Dataset | Notes |
|---|---|---|---|
| Q8 | Which hand is eating? | KU Leuven, both wrists | needs a model that reads both wrists together (12 channels) |

Left out on purpose:

- Q10, training first on Clemson to help bite detection: the two datasets differ in units, gravity and readings per second.
- Meal vs snack classification: mostly learns the time of day.
- Bite counts scored on Clemson: there are no bite labels to check against.
- Predicting BMI or identifying the person from motion: weak and ethically risky. Re-identification risk is discussed in the privacy section instead.

## Order of work

```text
1. Train and score the baselines on real data (Q1–Q3)
2. Architecture comparison (Q4)
3. New-user experiment (Q5) and watch efficiency with int8 (Q6)
4. Meal-detector breakdowns and daily totals (Q7, Q9)
5. Phone-to-model converter, tested on a recorded phone file
6. Live demo
7. Report, including the privacy design
(Q8 if time allows)
```

Requirements for the live demo:

- The demo model must not rely on future readings beyond a short delay. The CNN needs about 8 s of look-ahead. The bidirectional GRU and the transformer need a causal variant.
- The converter needs a setting per phone type. Android usually reports m/s² and rad/s. iPhone usually reports g and rad/s, with gravity provided separately.
- A short calibration pose at the start maps phone axes to training axes.

---

# Course Submission

Course: **CMPS 364 — Advanced Machine Learning**

Team members: add names here.

Submission deliverables:

- This README and reproducible preprocessing/training code
- Real-data results summary and rendered analysis notebook
- Poster-ready figures under `results/figures/`
- Live replay and Sensor Logger demonstration
- Report and poster, including the project privacy and limitation analysis

Any generative AI tool used in the project should be acknowledged according to the course policy.

# Datasets and Tools

- **KU Leuven FD-I:** public wrist-IMU eating and drinking gesture dataset used for gesture classification, bite counting, eating speed, personalization, and architecture comparison.
- **Clemson All-day:** public all-day wrist-motion and eating-event dataset used for meal detection and daily eating totals.
- **PyTorch:** model training, evaluation, live inference, and int8 quantization.
- **Sensor Logger:** mobile accelerometer, gravity, and gyroscope streaming and export tool used by the live demo.

Use the datasets only under their published terms. This repository does not redistribute raw KU Leuven or Clemson data.
