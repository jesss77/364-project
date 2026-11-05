import csv
import gc
import json
import pickle
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "kuleuven"
)

FD1_DIR = RAW_DIR / "FD-I"
FD1_ARCHIVE = RAW_DIR / "FD-I.zip"

SPLIT_PATH = (
    ROOT
    / "config"
    / "fd1_dev_split.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "kuleuven_fd1"
)

PARTICIPANT_OUTPUT_DIR = (
    OUTPUT_DIR
    / "participants"
)

WINDOWS_OUTPUT = (
    OUTPUT_DIR
    / "windows.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "results"
    / "data_audit"
    / "fd1_preprocessing_summary.json"
)


RAW_SAMPLE_RATE = 64
TARGET_SAMPLE_RATE = 16

DOWNSAMPLE_FACTOR = (
    RAW_SAMPLE_RATE
    // TARGET_SAMPLE_RATE
)

WINDOW_SECONDS = 60
STRIDE_SECONDS = 30

WINDOW_SAMPLES = (
    WINDOW_SECONDS
    * TARGET_SAMPLE_RATE
)

STRIDE_SAMPLES = (
    STRIDE_SECONDS
    * TARGET_SAMPLE_RATE
)

EXPECTED_PARTICIPANTS = 34
EXPECTED_CHANNELS = 6

WRISTS = (
    "L",
    "R",
)


def ensure_raw_files():
    required_files = [
        FD1_DIR / "X_L.pkl",
        FD1_DIR / "X_R.pkl",
        FD1_DIR / "Y_L.pkl",
        FD1_DIR / "Y_R.pkl",
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if not missing:
        return

    if not FD1_ARCHIVE.exists():
        raise FileNotFoundError(
            "FD-I files are missing and FD-I.zip "
            "was not found.\n\n"
            "Download it first with:\n"
            "python scripts/data/download_data.py "
            "--dataset kuleuven --file FD-I.zip"
        )

    print(
        "Some FD-I files are missing."
    )
    print(
        "Extracting them from FD-I.zip..."
    )

    with zipfile.ZipFile(
        FD1_ARCHIVE,
        "r",
    ) as archive:

        members = [
            "FD-I/X_L.pkl",
            "FD-I/X_R.pkl",
            "FD-I/Y_L.pkl",
            "FD-I/Y_R.pkl",
        ]

        archive.extractall(
            RAW_DIR,
            members=members,
        )

    print(
        "Extraction complete."
    )


def load_split():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: "
            f"{SPLIT_PATH}"
        )

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
            if (
                participant_id
                in participant_to_split
            ):
                raise ValueError(
                    f"Participant "
                    f"{participant_id} "
                    f"appears in multiple splits."
                )

            participant_to_split[
                participant_id
            ] = split_name

    expected = set(
        range(
            EXPECTED_PARTICIPANTS
        )
    )

    actual = set(
        participant_to_split
    )

    if actual != expected:
        raise ValueError(
            "Split does not contain exactly "
            "participants 0-33."
        )

    return (
        split,
        participant_to_split,
    )


def load_pickle(path):
    print(
        f"Loading {path.name}..."
    )

    with path.open(
        "rb"
    ) as file:
        data = pickle.load(
            file
        )

    print(
        f"Loaded {path.name}"
    )

    return data


def normalize_labels(labels):
    labels = np.asarray(
        labels
    ).reshape(-1)

    unexpected = (
        set(
            np.unique(
                labels
            ).tolist()
        )
        - {0, 1, 2}
    )

    if unexpected:
        raise ValueError(
            f"Unexpected labels: "
            f"{sorted(unexpected)}"
        )

    return labels.astype(
        np.uint8,
        copy=False,
    )


def downsample_sensor(sensor):
    sensor = np.asarray(
        sensor
    )

    if (
        sensor.ndim != 2
        or sensor.shape[1]
        != EXPECTED_CHANNELS
    ):
        raise ValueError(
            f"Unexpected sensor shape: "
            f"{sensor.shape}"
        )

    downsampled = resample_poly(
        sensor,
        up=1,
        down=DOWNSAMPLE_FACTOR,
        axis=0,
    )

    return downsampled.astype(
        np.float32
    )


def downsample_labels(labels):
    """
    Labels are already point-wise annotations.

    The new 16 Hz sample at index j corresponds
    to the timeline position of raw sample j * 4,
    so we preserve the label at that position.
    """

    return labels[
        ::DOWNSAMPLE_FACTOR
    ].astype(
        np.uint8,
        copy=False,
    )


def save_participant(
    participant_id,
    wrist,
    split_name,
    sensor,
    labels,
):
    split_dir = (
        PARTICIPANT_OUTPUT_DIR
        / split_name
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"participant_"
        f"{participant_id:02d}_"
        f"{wrist}.npz"
    )

    path = (
        split_dir
        / filename
    )

    np.savez(
        path,
        X=sensor,
        y=labels,
        participant_id=np.int16(
            participant_id
        ),
        wrist=np.array(
            wrist
        ),
        sample_rate_hz=np.int16(
            TARGET_SAMPLE_RATE
        ),
    )

    return path


def create_window_rows(
    participant_id,
    wrist,
    split_name,
    processed_path,
    labels,
):
    rows = []

    window_index = 0

    for start in range(
        0,
        len(labels)
        - WINDOW_SAMPLES
        + 1,
        STRIDE_SAMPLES,
    ):
        end = (
            start
            + WINDOW_SAMPLES
        )

        window_labels = labels[
            start:end
        ]

        counts = np.bincount(
            window_labels,
            minlength=3,
        )

        rows.append(
            {
                "split":
                    split_name,

                "participant_id":
                    participant_id,

                "wrist":
                    wrist,

                "file":
                    str(
                        processed_path
                        .relative_to(ROOT)
                    ),

                "window_index":
                    window_index,

                "start_sample":
                    start,

                "end_sample":
                    end,

                "start_seconds":
                    start
                    / TARGET_SAMPLE_RATE,

                "end_seconds":
                    end
                    / TARGET_SAMPLE_RATE,

                "other_samples":
                    int(
                        counts[0]
                    ),

                "eating_samples":
                    int(
                        counts[1]
                    ),

                "drinking_samples":
                    int(
                        counts[2]
                    ),

                "contains_eating":
                    int(
                        counts[1] > 0
                    ),

                "contains_drinking":
                    int(
                        counts[2] > 0
                    ),
            }
        )

        window_index += 1

    return rows


def process_wrist(
    wrist,
    participant_to_split,
):
    x_path = (
        FD1_DIR
        / f"X_{wrist}.pkl"
    )

    y_path = (
        FD1_DIR
        / f"Y_{wrist}.pkl"
    )

    x_data = load_pickle(
        x_path
    )

    y_data = load_pickle(
        y_path
    )

    if (
        len(x_data)
        != EXPECTED_PARTICIPANTS
    ):
        raise ValueError(
            f"X_{wrist} contains "
            f"{len(x_data)} participants."
        )

    if (
        len(y_data)
        != EXPECTED_PARTICIPANTS
    ):
        raise ValueError(
            f"Y_{wrist} contains "
            f"{len(y_data)} participants."
        )

    wrist_rows = []
    participant_summaries = []

    for participant_id in range(
        EXPECTED_PARTICIPANTS
    ):
        split_name = (
            participant_to_split[
                participant_id
            ]
        )

        raw_sensor = np.asarray(
            x_data[
                participant_id
            ]
        )

        raw_labels = (
            normalize_labels(
                y_data[
                    participant_id
                ]
            )
        )

        if (
            len(raw_sensor)
            != len(raw_labels)
        ):
            raise ValueError(
                f"Raw X/Y length mismatch "
                f"for participant "
                f"{participant_id}, "
                f"wrist {wrist}."
            )

        sensor = (
            downsample_sensor(
                raw_sensor
            )
        )

        labels = (
            downsample_labels(
                raw_labels
            )
        )

        # resample_poly and simple label
        # decimation should normally produce
        # the same length. Trim defensively
        # if an edge-rounding difference occurs.
        final_length = min(
            len(sensor),
            len(labels),
        )

        sensor = sensor[
            :final_length
        ]

        labels = labels[
            :final_length
        ]

        output_path = (
            save_participant(
                participant_id=
                    participant_id,

                wrist=
                    wrist,

                split_name=
                    split_name,

                sensor=
                    sensor,

                labels=
                    labels,
            )
        )

        window_rows = (
            create_window_rows(
                participant_id=
                    participant_id,

                wrist=
                    wrist,

                split_name=
                    split_name,

                processed_path=
                    output_path,

                labels=
                    labels,
            )
        )

        wrist_rows.extend(
            window_rows
        )

        label_counts = np.bincount(
            labels,
            minlength=3,
        )

        participant_summaries.append(
            {
                "participant_id":
                    participant_id,

                "wrist":
                    wrist,

                "split":
                    split_name,

                "raw_samples":
                    int(
                        len(
                            raw_sensor
                        )
                    ),

                "processed_samples":
                    int(
                        final_length
                    ),

                "processed_hours":
                    float(
                        final_length
                        / TARGET_SAMPLE_RATE
                        / 3600
                    ),

                "other_samples":
                    int(
                        label_counts[0]
                    ),

                "eating_samples":
                    int(
                        label_counts[1]
                    ),

                "drinking_samples":
                    int(
                        label_counts[2]
                    ),

                "windows":
                    len(
                        window_rows
                    ),
            }
        )

        print(
            f"{wrist} "
            f"participant "
            f"{participant_id:02d} "
            f"[{split_name:<10}] "
            f"{len(raw_sensor):,} "
            f"→ "
            f"{final_length:,} samples, "
            f"{len(window_rows):,} windows"
        )

        del raw_sensor
        del raw_labels
        del sensor
        del labels

    del x_data
    del y_data

    gc.collect()

    return (
        wrist_rows,
        participant_summaries,
    )


def write_windows(rows):
    WINDOWS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "split",
        "participant_id",
        "wrist",
        "file",
        "window_index",
        "start_sample",
        "end_sample",
        "start_seconds",
        "end_seconds",
        "other_samples",
        "eating_samples",
        "drinking_samples",
        "contains_eating",
        "contains_drinking",
    ]

    with WINDOWS_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def build_summary(
    split,
    participant_summaries,
    window_rows,
):
    summary = {
        "dataset":
            "KU Leuven FD-I",

        "raw_sample_rate_hz":
            RAW_SAMPLE_RATE,

        "processed_sample_rate_hz":
            TARGET_SAMPLE_RATE,

        "downsample_factor":
            DOWNSAMPLE_FACTOR,

        "sensor_resampling":
            "scipy.signal.resample_poly",

        "label_resampling":
            "take every 4th point-wise label",

        "wrists":
            "left and right processed separately",

        "window_seconds":
            WINDOW_SECONDS,

        "stride_seconds":
            STRIDE_SECONDS,

        "overlap":
            0.50,

        "window_samples":
            WINDOW_SAMPLES,

        "development_split":
            split,

        "participant_files":
            len(
                participant_summaries
            ),

        "total_windows":
            len(
                window_rows
            ),

        "splits": {},
    }

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_participants = [
            row
            for row
            in participant_summaries
            if row[
                "split"
            ]
            == split_name
        ]

        split_windows = [
            row
            for row
            in window_rows
            if row[
                "split"
            ]
            == split_name
        ]

        wrist_counts = Counter(
            row[
                "wrist"
            ]
            for row
            in split_windows
        )

        summary[
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

            "participant_wrist_files":
                len(
                    split_participants
                ),

            "windows":
                len(
                    split_windows
                ),

            "left_windows":
                wrist_counts[
                    "L"
                ],

            "right_windows":
                wrist_counts[
                    "R"
                ],

            "windows_with_eating":
                sum(
                    int(
                        row[
                            "contains_eating"
                        ]
                    )
                    for row
                    in split_windows
                ),

            "windows_with_drinking":
                sum(
                    int(
                        row[
                            "contains_drinking"
                        ]
                    )
                    for row
                    in split_windows
                ),
        }

    return summary


def main():
    print()
    print(
        "KU LEUVEN FD-I PREPROCESSING"
    )
    print("=" * 60)

    print(
        f"Raw rate:        "
        f"{RAW_SAMPLE_RATE} Hz"
    )

    print(
        f"Output rate:     "
        f"{TARGET_SAMPLE_RATE} Hz"
    )

    print(
        f"Window:          "
        f"{WINDOW_SECONDS} s "
        f"({WINDOW_SAMPLES} samples)"
    )

    print(
        f"Stride:          "
        f"{STRIDE_SECONDS} s "
        f"({STRIDE_SAMPLES} samples)"
    )

    print(
        "Wrists:          separate"
    )

    print()

    ensure_raw_files()

    (
        split,
        participant_to_split,
    ) = load_split()

    all_window_rows = []
    all_participant_summaries = []

    for wrist in WRISTS:
        print()
        print(
            f"PROCESSING WRIST {wrist}"
        )
        print("-" * 60)

        (
            window_rows,
            participant_summaries,
        ) = process_wrist(
            wrist=
                wrist,

            participant_to_split=
                participant_to_split,
        )

        all_window_rows.extend(
            window_rows
        )

        all_participant_summaries.extend(
            participant_summaries
        )

    write_windows(
        all_window_rows
    )

    summary = build_summary(
        split=
            split,

        participant_summaries=
            all_participant_summaries,

        window_rows=
            all_window_rows,
    )

    AUDIT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_OUTPUT.open(
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
        "PREPROCESSING COMPLETE"
    )
    print("=" * 60)

    print(
        f"Participant/wrist files: "
        f"{summary['participant_files']}"
    )

    print(
        f"Total windows:            "
        f"{summary['total_windows']:,}"
    )

    print()

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        details = (
            summary[
                "splits"
            ][
                split_name
            ]
        )

        print(
            f"{split_name.upper()}"
        )

        print(
            f"  participants:          "
            f"{details['participants']}"
        )

        print(
            f"  windows:               "
            f"{details['windows']:,}"
        )

        print(
            f"  windows with eating:   "
            f"{details['windows_with_eating']:,}"
        )

        print(
            f"  windows with drinking: "
            f"{details['windows_with_drinking']:,}"
        )

    print()

    print(
        "Saved processed participants under:"
    )

    print(
        f"  "
        f"{PARTICIPANT_OUTPUT_DIR.relative_to(ROOT)}"
    )

    print()

    print(
        "Saved window index:"
    )

    print(
        f"  "
        f"{WINDOWS_OUTPUT.relative_to(ROOT)}"
    )

    print()

    print(
        "Saved audit summary:"
    )

    print(
        f"  "
        f"{AUDIT_OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()