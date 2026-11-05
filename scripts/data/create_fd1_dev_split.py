import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    ROOT
    / "config"
    / "fd1_dev_split.json"
)

SEED = 42
PARTICIPANT_COUNT = 34

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.20


def main():
    participant_ids = np.arange(
        PARTICIPANT_COUNT
    )

    rng = np.random.default_rng(
        SEED
    )

    shuffled = rng.permutation(
        participant_ids
    )

    train_count = round(
        PARTICIPANT_COUNT
        * TRAIN_RATIO
    )

    validation_count = round(
        PARTICIPANT_COUNT
        * VALIDATION_RATIO
    )

    train = shuffled[
        :train_count
    ]

    validation = shuffled[
        train_count:
        train_count + validation_count
    ]

    test = shuffled[
        train_count + validation_count:
    ]

    # Sort within each group only to make
    # the saved JSON easier to read.
    train = sorted(
        train.tolist()
    )

    validation = sorted(
        validation.tolist()
    )

    test = sorted(
        test.tolist()
    )

    # Safety checks
    all_ids = (
        train
        + validation
        + test
    )

    if len(all_ids) != PARTICIPANT_COUNT:
        raise ValueError(
            "Participant count is incorrect."
        )

    if len(set(all_ids)) != PARTICIPANT_COUNT:
        raise ValueError(
            "A participant appears in more than one split."
        )

    if set(all_ids) != set(
        range(PARTICIPANT_COUNT)
    ):
        raise ValueError(
            "Some participants are missing."
        )

    split = {
        "dataset": "KU Leuven FD-I",
        "purpose": "Development split",
        "seed": SEED,
        "split_by": "participant",
        "ratios": {
            "train": 0.70,
            "validation": 0.20,
            "test": 0.10,
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

    print(
        "KU LEUVEN FD-I DEVELOPMENT SPLIT"
    )
    print("=" * 50)

    print(
        f"Seed:       {SEED}"
    )

    print(
        f"Train:      {len(train)} participants"
    )
    print(
        f"Validation: {len(validation)} participants"
    )
    print(
        f"Test:       {len(test)} participants"
    )

    print()

    print(
        f"Train IDs:      {train}"
    )
    print(
        f"Validation IDs: {validation}"
    )
    print(
        f"Test IDs:       {test}"
    )

    print()

    print(
        "No participant overlap: YES"
    )

    print(
        f"Saved: {OUTPUT_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()