import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    ROOT
    / "data"
    / "processed"
    / "clemson_allday"
    / "recordings"
    / "train"
)

OUTPUT_PATH = (
    ROOT
    / "config"
    / "clemson_normalization.json"
)

SAMPLE_RATE = 15
CHANNELS = 6

CHANNEL_NAMES = [
    "linear_acc_x",
    "linear_acc_y",
    "linear_acc_z",
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",
]


def main():
    print()
    print("CLEMSON ALL-DAY NORMALIZATION")
    print("=" * 60)

    files = sorted(TRAIN_DIR.glob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No training files found in {TRAIN_DIR.relative_to(ROOT)}.\n"
            "Run scripts/data/preprocess_clemson.py first."
        )

    print(f"Training files: {len(files)}")

    total_samples = 0
    skipped_samples = 0

    channel_sum = np.zeros(CHANNELS, dtype=np.float64)
    channel_squared_sum = np.zeros(CHANNELS, dtype=np.float64)

    for index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as data:
            X = data["X"]

        if X.ndim != 2 or X.shape[1] != CHANNELS:
            raise ValueError(f"{path.name}: unexpected shape {X.shape}")

        finite = np.isfinite(X).all(axis=1)
        skipped_samples += int((~finite).sum())

        X = X[finite].astype(np.float64)

        channel_sum += X.sum(axis=0)
        channel_squared_sum += (X * X).sum(axis=0)
        total_samples += len(X)

        print(
            f"[{index:03d}/{len(files):03d}] "
            f"{path.name}: {len(X):,} samples"
        )

    mean = channel_sum / total_samples
    variance = np.maximum(
        channel_squared_sum / total_samples - mean ** 2,
        0,
    )
    std = np.sqrt(variance)

    if np.any(std == 0):
        raise ValueError("At least one channel has standard deviation 0.")

    output = {
        "dataset": "Clemson All-day",
        "calculated_from": "training recordings only",
        "sample_rate_hz": SAMPLE_RATE,
        "channels": CHANNEL_NAMES,
        "training_files": len(files),
        "training_samples": total_samples,
        "skipped_nonfinite_samples": skipped_samples,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print()
    print("NORMALIZATION VALUES")
    print("=" * 60)

    for channel, name in enumerate(CHANNEL_NAMES):
        print(
            f"{name:<13} mean = {mean[channel]:.6f}, "
            f"std = {std[channel]:.6f}"
        )

    print()
    print(f"Training samples used: {total_samples:,}")
    print(f"Skipped NaN/Inf rows:  {skipped_samples:,}")
    print()
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
