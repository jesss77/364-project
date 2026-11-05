import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "kuleuven_fd1"
)

PARTICIPANT_DIR = (
    PROCESSED_DIR
    / "participants"
)

WINDOWS_PATH = (
    PROCESSED_DIR
    / "windows.csv"
)

SPLIT_PATH = (
    ROOT
    / "config"
    / "fd1_dev_split.json"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "data_audit"
    / "fd1_processed_validation.json"
)


EXPECTED_PARTICIPANTS = 34
EXPECTED_WRISTS = {"L", "R"}

EXPECTED_SAMPLE_RATE = 16
EXPECTED_CHANNELS = 6

WINDOW_SAMPLES = 960


def load_split():
    with SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        split = json.load(file)

    participant_to_split = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        for participant_id in split[
            split_name
        ]:
            participant_to_split[
                participant_id
            ] = split_name

    return split, participant_to_split


def validate_participant_files(
    participant_to_split,
):
    files = sorted(
        PARTICIPANT_DIR.rglob(
            "*.npz"
        )
    )

    expected_file_count = (
        EXPECTED_PARTICIPANTS
        * len(
            EXPECTED_WRISTS
        )
    )

    if len(files) != expected_file_count:
        raise ValueError(
            f"Expected "
            f"{expected_file_count} "
            f"participant/wrist files, "
            f"found {len(files)}."
        )

    summary = defaultdict(
        lambda: {
            "files": 0,
            "samples": 0,
            "label_counts": Counter(),
        }
    )

    participant_wrists = defaultdict(
        set
    )

    participant_lengths = defaultdict(
        dict
    )

    errors = []

    for path in files:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            X = data["X"]
            y = data["y"]

            participant_id = int(
                data[
                    "participant_id"
                ]
            )

            wrist = str(
                data[
                    "wrist"
                ]
            )

            sample_rate = int(
                data[
                    "sample_rate_hz"
                ]
            )

        split_name = (
            participant_to_split.get(
                participant_id
            )
        )

        if split_name is None:
            errors.append(
                f"{path.name}: "
                f"unknown participant "
                f"{participant_id}"
            )
            continue

        expected_split_dir = (
            PARTICIPANT_DIR
            / split_name
        )

        if path.parent != expected_split_dir:
            errors.append(
                f"{path.name}: "
                f"stored in wrong split folder"
            )

        if wrist not in EXPECTED_WRISTS:
            errors.append(
                f"{path.name}: "
                f"unexpected wrist {wrist}"
            )

        if X.ndim != 2:
            errors.append(
                f"{path.name}: "
                f"X must be 2D, "
                f"got {X.shape}"
            )

        elif X.shape[1] != EXPECTED_CHANNELS:
            errors.append(
                f"{path.name}: "
                f"expected "
                f"{EXPECTED_CHANNELS} "
                f"channels, got "
                f"{X.shape[1]}"
            )

        if len(X) != len(y):
            errors.append(
                f"{path.name}: "
                f"X/Y length mismatch"
            )

        if X.dtype != np.float32:
            errors.append(
                f"{path.name}: "
                f"X dtype is "
                f"{X.dtype}, "
                f"expected float32"
            )

        if y.dtype != np.uint8:
            errors.append(
                f"{path.name}: "
                f"y dtype is "
                f"{y.dtype}, "
                f"expected uint8"
            )

        if sample_rate != EXPECTED_SAMPLE_RATE:
            errors.append(
                f"{path.name}: "
                f"sample rate is "
                f"{sample_rate}, "
                f"expected "
                f"{EXPECTED_SAMPLE_RATE}"
            )

        if not np.isfinite(
            X
        ).all():
            errors.append(
                f"{path.name}: "
                f"X contains NaN/Inf"
            )

        labels = set(
            np.unique(
                y
            ).tolist()
        )

        unexpected_labels = (
            labels
            - {0, 1, 2}
        )

        if unexpected_labels:
            errors.append(
                f"{path.name}: "
                f"unexpected labels "
                f"{sorted(unexpected_labels)}"
            )

        counts = np.bincount(
            y,
            minlength=3,
        )

        summary[
            split_name
        ][
            "files"
        ] += 1

        summary[
            split_name
        ][
            "samples"
        ] += len(y)

        summary[
            split_name
        ][
            "label_counts"
        ].update(
            {
                0: int(
                    counts[0]
                ),
                1: int(
                    counts[1]
                ),
                2: int(
                    counts[2]
                ),
            }
        )

        participant_wrists[
            participant_id
        ].add(
            wrist
        )

        participant_lengths[
            participant_id
        ][
            wrist
        ] = len(y)

    for participant_id in range(
        EXPECTED_PARTICIPANTS
    ):
        wrists = (
            participant_wrists[
                participant_id
            ]
        )

        if wrists != EXPECTED_WRISTS:
            errors.append(
                f"Participant "
                f"{participant_id}: "
                f"wrists found "
                f"{sorted(wrists)}"
            )

        lengths = (
            participant_lengths[
                participant_id
            ]
        )

        if (
            "L" in lengths
            and "R" in lengths
            and lengths["L"]
            != lengths["R"]
        ):
            errors.append(
                f"Participant "
                f"{participant_id}: "
                f"left/right lengths differ"
            )

    return summary, errors


def validate_windows():
    errors = []

    counts = Counter()

    previous = {}

    with WINDOWS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            split_name = row[
                "split"
            ]

            participant_id = int(
                row[
                    "participant_id"
                ]
            )

            wrist = row[
                "wrist"
            ]

            start = int(
                row[
                    "start_sample"
                ]
            )

            end = int(
                row[
                    "end_sample"
                ]
            )

            length = (
                end
                - start
            )

            if length != WINDOW_SAMPLES:
                errors.append(
                    f"windows.csv row "
                    f"{row_number}: "
                    f"window length "
                    f"{length}"
                )

            key = (
                split_name,
                participant_id,
                wrist,
            )

            if key in previous:
                difference = (
                    start
                    - previous[key]
                )

                if difference != 480:
                    errors.append(
                        f"windows.csv row "
                        f"{row_number}: "
                        f"stride is "
                        f"{difference}, "
                        f"expected 480"
                    )

            previous[
                key
            ] = start

            counts[
                split_name
            ] += 1

    return counts, errors


def main():
    print()
    print(
        "KU LEUVEN PROCESSED DATA VALIDATION"
    )
    print("=" * 60)

    (
        split,
        participant_to_split,
    ) = load_split()

    (
        sample_summary,
        participant_errors,
    ) = validate_participant_files(
        participant_to_split
    )

    (
        window_counts,
        window_errors,
    ) = validate_windows()

    errors = (
        participant_errors
        + window_errors
    )

    output = {
        "dataset":
            "KU Leuven FD-I",

        "sample_rate_hz":
            EXPECTED_SAMPLE_RATE,

        "channels":
            EXPECTED_CHANNELS,

        "window_samples":
            WINDOW_SAMPLES,

        "splits": {},

        "errors":
            errors,
    }

    total_samples = 0
    total_labels = Counter()

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        data = (
            sample_summary[
                split_name
            ]
        )

        labels = (
            data[
                "label_counts"
            ]
        )

        samples = (
            data[
                "samples"
            ]
        )

        total_samples += samples

        total_labels.update(
            labels
        )

        percentages = {}

        for label in (
            0,
            1,
            2,
        ):
            count = labels[
                label
            ]

            if samples:
                percentage = (
                    count
                    / samples
                    * 100
                )
            else:
                percentage = 0

            percentages[
                str(label)
            ] = percentage

        output[
            "splits"
        ][
            split_name
        ] = {
            "participants":
                len(
                    split[
                        split_name
                    ]
                ),

            "files":
                data[
                    "files"
                ],

            "samples":
                samples,

            "windows":
                window_counts[
                    split_name
                ],

            "labels": {
                "0_other": int(
                    labels[0]
                ),

                "1_eating": int(
                    labels[1]
                ),

                "2_drinking": int(
                    labels[2]
                ),
            },

            "percentages": {
                "0_other":
                    percentages["0"],

                "1_eating":
                    percentages["1"],

                "2_drinking":
                    percentages["2"],
            },
        }

    output[
        "total_samples"
    ] = total_samples

    output[
        "total_labels"
    ] = {
        "0_other": int(
            total_labels[0]
        ),

        "1_eating": int(
            total_labels[1]
        ),

        "2_drinking": int(
            total_labels[2]
        ),
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

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        data = output[
            "splits"
        ][
            split_name
        ]

        print()
        print(
            split_name.upper()
        )

        print(
            f"  participants: "
            f"{data['participants']}"
        )

        print(
            f"  files:        "
            f"{data['files']}"
        )

        print(
            f"  windows:      "
            f"{data['windows']:,}"
        )

        print(
            f"  samples:      "
            f"{data['samples']:,}"
        )

        print(
            f"  other:        "
            f"{data['labels']['0_other']:,} "
            f"("
            f"{data['percentages']['0_other']:.2f}%"
            f")"
        )

        print(
            f"  eating:       "
            f"{data['labels']['1_eating']:,} "
            f"("
            f"{data['percentages']['1_eating']:.2f}%"
            f")"
        )

        print(
            f"  drinking:     "
            f"{data['labels']['2_drinking']:,} "
            f"("
            f"{data['percentages']['2_drinking']:.2f}%"
            f")"
        )

    print()
    print("=" * 60)

    if errors:
        print(
            f"VALIDATION FAILED: "
            f"{len(errors)} issue(s)"
        )

        for error in errors:
            print(
                f"  - {error}"
            )

        raise SystemExit(1)

    print(
        "VALIDATION PASSED"
    )

    print(
        "All processed files, labels, "
        "splits and windows are consistent."
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