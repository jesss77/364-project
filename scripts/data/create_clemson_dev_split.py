import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RECORDINGS_PATH = (
    ROOT
    / "results"
    / "data_audit"
    / "clemson_recording_alignment.csv"
)

OUTPUT_PATH = (
    ROOT
    / "config"
    / "clemson_dev_split.json"
)

SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def load_recording_ids():
    if not RECORDINGS_PATH.exists():
        raise FileNotFoundError(
            f"{RECORDINGS_PATH.relative_to(ROOT)} not found.\n"
            "Run scripts/data/validate_clemson_alignment.py first."
        )

    with RECORDINGS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        ids = [
            row["recording_id"]
            for row in csv.DictReader(file)
        ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate recording IDs in alignment file."
        )

    return sorted(ids)


def main():
    recording_ids = load_recording_ids()
    count = len(recording_ids)

    rng = np.random.default_rng(SEED)

    shuffled = [
        recording_ids[index]
        for index in rng.permutation(count)
    ]

    train_count = round(count * TRAIN_RATIO)
    validation_count = round(count * VALIDATION_RATIO)

    train = sorted(shuffled[:train_count])
    validation = sorted(
        shuffled[train_count:train_count + validation_count]
    )
    test = sorted(shuffled[train_count + validation_count:])

    all_ids = train + validation + test

    if len(all_ids) != count or len(set(all_ids)) != count:
        raise ValueError(
            "A recording is missing or appears in more than one split."
        )

    if set(all_ids) != set(recording_ids):
        raise ValueError(
            "Split IDs do not match the alignment file."
        )

    split = {
        "dataset": "Clemson All-day",
        "purpose": "Development split",
        "seed": SEED,
        "split_by": "recording_id (one recording per participant ID)",
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": round(1 - TRAIN_RATIO - VALIDATION_RATIO, 2),
        },
        "train": train,
        "validation": validation,
        "test": test,
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
            split,
            file,
            indent=2,
        )

    print("CLEMSON ALL-DAY DEVELOPMENT SPLIT")
    print("=" * 50)
    print(f"Seed:       {SEED}")
    print(f"Recordings: {count}")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")
    print()
    print("No recording overlap: YES")
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
