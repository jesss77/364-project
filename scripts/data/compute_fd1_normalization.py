import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PARTICIPANT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "kuleuven_fd1"
    / "participants"
)

TRAIN_DIR = (
    PARTICIPANT_DIR
    / "train"
)

OUTPUT_PATH = (
    ROOT
    / "config"
    / "fd1_normalization.json"
)

EXPECTED_CHANNELS = 6


def main():
    print()
    print(
        "KU LEUVEN FD-I NORMALIZATION"
    )
    print("=" * 60)

    files = sorted(
        TRAIN_DIR.glob(
            "*.npz"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No training files found in "
            f"{TRAIN_DIR}"
        )

    print(
        f"Training files: {len(files)}"
    )

    total_samples = 0

    channel_sum = np.zeros(
        EXPECTED_CHANNELS,
        dtype=np.float64,
    )

    channel_squared_sum = np.zeros(
        EXPECTED_CHANNELS,
        dtype=np.float64,
    )

    for index, path in enumerate(
        files,
        start=1,
    ):
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            X = data["X"]

        if (
            X.ndim != 2
            or X.shape[1]
            != EXPECTED_CHANNELS
        ):
            raise ValueError(
                f"{path.name}: "
                f"unexpected shape "
                f"{X.shape}"
            )

        X = X.astype(
            np.float64,
            copy=False,
        )

        channel_sum += np.sum(
            X,
            axis=0,
        )

        channel_squared_sum += np.sum(
            X * X,
            axis=0,
        )

        total_samples += len(
            X
        )

        print(
            f"[{index:02d}/{len(files):02d}] "
            f"{path.name}: "
            f"{len(X):,} samples"
        )

    mean = (
        channel_sum
        / total_samples
    )

    variance = (
        channel_squared_sum
        / total_samples
        - mean ** 2
    )

    variance = np.maximum(
        variance,
        0,
    )

    std = np.sqrt(
        variance
    )

    if np.any(
        std == 0
    ):
        raise ValueError(
            "At least one channel has "
            "standard deviation 0."
        )

    output = {
        "dataset":
            "KU Leuven FD-I",

        "calculated_from":
            "training participants only",

        "sample_rate_hz":
            16,

        "channels": [
            "channel_0",
            "channel_1",
            "channel_2",
            "channel_3",
            "channel_4",
            "channel_5",
        ],

        "training_files":
            len(files),

        "training_samples":
            total_samples,

        "mean": [
            float(value)
            for value in mean
        ],

        "std": [
            float(value)
            for value in std
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print(
        "NORMALIZATION VALUES"
    )
    print("=" * 60)

    for channel in range(
        EXPECTED_CHANNELS
    ):
        print(
            f"Channel {channel}: "
            f"mean = "
            f"{mean[channel]:.6f}, "
            f"std = "
            f"{std[channel]:.6f}"
        )

    print()
    print(
        f"Training samples used: "
        f"{total_samples:,}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"  "
        f"{OUTPUT_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()