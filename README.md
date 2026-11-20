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

---

# Current Phase

## Phase 1 — Dataset Validation and Reproducible Data Pipeline

The current goal is to make the complete data pipeline reproducible for all group members before model development begins.

Phase 1 includes:

1. Downloading both datasets from their official sources.
2. Verifying downloaded files and checksums.
3. Inspecting the actual released data formats.
4. Parsing and validating labels.
5. Aligning labels with sensor recordings.
6. Creating deterministic processed outputs.
7. Defining participant-independent train/validation/test splits.
8. Making sure another teammate can reproduce everything from a clean clone.

Model development starts only after this phase is complete.

All Phase 1 scripts now exist. KU Leuven is preprocessed and validated. The Clemson model-ready preprocessing scripts are written but have not yet been run on the full `SHMfiles.zip`. See [Phase 1 Status](#phase-1-status).

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
│   └── clemson_normalization.json      (created by compute_clemson_normalization.py)
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
│   └── train/
│       ├── common.py
│       ├── models.py
│       ├── metrics.py
│       ├── train_fd1.py
│       ├── train_clemson.py
│       └── compare_fd1_runs.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Later phases will add separate folders for model training and evaluation.

---

# Setup

## 1. Clone the repository

```powershell
git clone https://github.com/jesss77/364-project.git
cd 364-project
```

If working on the Phase 1 branch:

```powershell
git switch data-acquisition
```

---

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

Current Phase 1 dependencies include:

- `numpy`
- `scipy`
- `openpyxl`

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

The 7 official FD-I folds (`FD1_7_fold_id_list.zip`) are still available for cross-validation later. The development split above is used for model development.

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

These offsets will later be mapped to sensor samples.

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
- Clemson model-ready preprocessing, validation and normalization scripts

## Remaining

- Run `preprocess_clemson.py`, `validate_clemson_processed.py` and `compute_clemson_normalization.py` on the full `SHMfiles.zip`, then commit `clemson_normalization.json` and the two audit files
- Check the recordings whose log is about a day longer than the sensor data (`P2118`, `P2154`)
- Merge `data-acquisition` into `main`

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
common.py         paths, seeding, device choice, window dataset, full-recording inference
models.py         KU Leuven gesture models (cnn, cnn_gru, transformer) and EpisodeCNN (Clemson)
metrics.py        confusion matrix, precision/recall/F1, segment extraction and matching
train_fd1.py      KU Leuven gesture model (--model picks the architecture)
train_clemson.py  Clemson episode model
compare_fd1_runs.py  table of all KU Leuven runs → results/fd1_model_comparison.csv
```

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
- Imbalance: weighted cross-entropy. By default each class weight is the square root of its inverse frequency (`--class-weight sqrt|inverse|none`).
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

## Test split

Neither script scores the test split unless `--eval-test` is given. Use it only for the final run of a model that will be reported. All tuning uses validation only.

## Quick check

For a short run, for example on a laptop, add `--epochs 2 --max-train-windows 2000`.

---

# Next Phase

The first goal is to obtain working baselines for:

```text
Clemson All-day
    → eating-episode classifier
    → episode reconstruction

KU Leuven FD-I
    → eating-gesture classifier
    → bite-event detection
    → bite-count estimation
```

After the baselines work reliably, the project will move to:

```text
CNN vs CNN-GRU vs small Transformer   (code ready, see above)
↓
personalization
↓
efficiency evaluation
↓
final testing and report
```