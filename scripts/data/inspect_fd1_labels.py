import csv
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "kuleuven" / "FD-I"
OUTPUT_DIR = ROOT / "results" / "data_audit"

SAMPLING_RATE = 64

LABEL_NAMES = {
    0: "other",
    1: "eating",
    2: "drinking",
}


def load_pickle(path):
    print(
        f"Loading {path.relative_to(ROOT)}..."
    )

    with path.open("rb") as file:
        data = pickle.load(file)

    print(
        f"Loaded {len(data)} participants."
    )

    return data


def count_runs(labels, target):
    mask = labels == target

    if len(mask) == 0:
        return 0

    starts = mask & np.concatenate(
        ([True], ~mask[:-1])
    )

    return int(
        starts.sum()
    )


def inspect_side(side):
    path = DATA_DIR / f"Y_{side}.pkl"

    participants = load_pickle(
        path
    )

    rows = []

    for participant_id, participant_labels in enumerate(
        participants
    ):
        labels = np.asarray(
            participant_labels
        ).reshape(-1)

        unique_labels, counts = np.unique(
            labels,
            return_counts=True,
        )

        label_counts = {
            int(label): int(count)
            for label, count in zip(
                unique_labels,
                counts,
            )
        }

        unexpected = sorted(
            set(label_counts)
            - set(LABEL_NAMES)
        )

        rows.append(
            {
                "participant_id":
                    participant_id,

                "side":
                    side,

                "samples":
                    int(
                        len(labels)
                    ),

                "duration_seconds":
                    len(labels)
                    / SAMPLING_RATE,

                "duration_hours":
                    len(labels)
                    / SAMPLING_RATE
                    / 3600,

                "other_samples":
                    label_counts.get(
                        0,
                        0,
                    ),

                "eating_samples":
                    label_counts.get(
                        1,
                        0,
                    ),

                "drinking_samples":
                    label_counts.get(
                        2,
                        0,
                    ),

                "eating_runs":
                    count_runs(
                        labels,
                        1,
                    ),

                "drinking_runs":
                    count_runs(
                        labels,
                        2,
                    ),

                "unexpected_labels":
                    unexpected,
            }
        )

    return rows


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    left_rows = inspect_side(
        "L"
    )

    right_rows = inspect_side(
        "R"
    )

    if (
        len(left_rows)
        != len(right_rows)
    ):
        raise ValueError(
            "Left and right files contain "
            "different participant counts."
        )

    participant_rows = []

    for left, right in zip(
        left_rows,
        right_rows,
    ):
        if (
            left["participant_id"]
            != right["participant_id"]
        ):
            raise ValueError(
                "Participant ordering differs "
                "between hands."
            )

        participant_rows.append(
            {
                "participant_id":
                    left[
                        "participant_id"
                    ],

                "left_samples":
                    left[
                        "samples"
                    ],

                "right_samples":
                    right[
                        "samples"
                    ],

                "sample_count_match":
                    (
                        left["samples"]
                        == right["samples"]
                    ),

                "duration_hours":
                    left[
                        "duration_hours"
                    ],

                "left_eating_runs":
                    left[
                        "eating_runs"
                    ],

                "right_eating_runs":
                    right[
                        "eating_runs"
                    ],

                "left_drinking_runs":
                    left[
                        "drinking_runs"
                    ],

                "right_drinking_runs":
                    right[
                        "drinking_runs"
                    ],

                "total_eating_runs":
                    (
                        left[
                            "eating_runs"
                        ]
                        + right[
                            "eating_runs"
                        ]
                    ),

                "total_drinking_runs":
                    (
                        left[
                            "drinking_runs"
                        ]
                        + right[
                            "drinking_runs"
                        ]
                    ),
            }
        )

    total_hours = sum(
        row[
            "duration_hours"
        ]
        for row in participant_rows
    )

    total_eating_runs = sum(
        row[
            "total_eating_runs"
        ]
        for row in participant_rows
    )

    total_drinking_runs = sum(
        row[
            "total_drinking_runs"
        ]
        for row in participant_rows
    )

    mismatched_lengths = [
        row[
            "participant_id"
        ]
        for row in participant_rows
        if not row[
            "sample_count_match"
        ]
    ]

    unexpected_labels = sorted(
        {
            label
            for row in (
                left_rows
                + right_rows
            )
            for label in row[
                "unexpected_labels"
            ]
        }
    )

    summary = {
        "dataset":
            "KU Leuven FD-I",

        "sampling_rate_hz":
            SAMPLING_RATE,

        "participants":
            len(
                participant_rows
            ),

        "total_hours":
            total_hours,

        "total_eating_runs":
            total_eating_runs,

        "total_drinking_runs":
            total_drinking_runs,

        "participants_with_left_right_length_mismatch":
            mismatched_lengths,

        "unexpected_labels":
            unexpected_labels,

        "readme_reference": {
            "participants":
                34,

            "hours":
                251.70,

            "eating_gestures":
                4568,

            "drinking_gestures":
                1100,
        },
    }

    csv_path = (
        OUTPUT_DIR
        / "fd1_participants.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=participant_rows[
                0
            ].keys(),
        )

        writer.writeheader()

        writer.writerows(
            participant_rows
        )

    json_path = (
        OUTPUT_DIR
        / "fd1_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print(
        "FD-I LABEL AUDIT"
    )
    print("=" * 50)

    print(
        f"Participants:     "
        f"{summary['participants']}"
    )

    print(
        f"Total hours:      "
        f"{summary['total_hours']:.2f}"
    )

    print(
        f"Eating runs:      "
        f"{summary['total_eating_runs']}"
    )

    print(
        f"Drinking runs:    "
        f"{summary['total_drinking_runs']}"
    )

    print(
        "Length mismatch:  "
        f"{summary['participants_with_left_right_length_mismatch']}"
    )

    print(
        f"Unknown labels:   "
        f"{summary['unexpected_labels']}"
    )

    print()

    print(
        f"Saved: "
        f"{csv_path.relative_to(ROOT)}"
    )

    print(
        f"Saved: "
        f"{json_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()