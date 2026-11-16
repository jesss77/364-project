import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SENSOR_ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "clemson_allday"
    / "SHMfiles.zip"
)

EVENTS_CSV = (
    ROOT
    / "data"
    / "processed"
    / "clemson_allday"
    / "events.csv"
)

SPLIT_PATH = (
    ROOT
    / "config"
    / "clemson_dev_split.json"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "clemson_allday"
)

RECORDING_OUTPUT_DIR = (
    OUTPUT_DIR
    / "recordings"
)

WINDOWS_OUTPUT = (
    OUTPUT_DIR
    / "windows.csv"
)

AUDIT_OUTPUT = (
    ROOT
    / "results"
    / "data_audit"
    / "clemson_preprocessing_summary.json"
)

SAMPLE_RATE = 15
CHANNELS = 6

WINDOW_SECONDS = 360
STRIDE_SECONDS = 180

WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE
STRIDE_SAMPLES = STRIDE_SECONDS * SAMPLE_RATE

ID_PATTERN = re.compile(
    r"\bP\d{4}\b",
    re.IGNORECASE,
)

WINDOW_FIELDS = [
    "split",
    "recording_id",
    "file",
    "window_index",
    "start_sample",
    "end_sample",
    "start_seconds",
    "end_seconds",
    "other_samples",
    "eating_samples",
    "contains_eating",
]


def normalize_id(value):
    match = ID_PATTERN.search(str(value))

    if match is None:
        return None

    return match.group(0).upper()


def load_split():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"{SPLIT_PATH.relative_to(ROOT)} not found.\n"
            "Run scripts/data/create_clemson_dev_split.py first."
        )

    with SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        split = json.load(file)

    recording_to_split = {}

    for split_name in ("train", "validation", "test"):
        for recording_id in split[split_name]:
            if recording_id in recording_to_split:
                raise ValueError(
                    f"{recording_id} appears in multiple splits."
                )

            recording_to_split[recording_id] = split_name

    return split, recording_to_split


def load_events():
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(
            f"{EVENTS_CSV.relative_to(ROOT)} not found.\n"
            "Run scripts/data/preprocess_clemson_events.py first."
        )

    events = {}

    with EVENTS_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            events.setdefault(
                row["recording_id"],
                [],
            ).append(
                (
                    float(row["start_offset_seconds"]),
                    float(row["end_offset_seconds"]),
                )
            )

    return events


def build_labels(
    sample_count,
    recording_events,
):
    labels = np.zeros(
        sample_count,
        dtype=np.uint8,
    )

    clipped = 0
    dropped = 0

    for start_seconds, end_seconds in recording_events:
        start = round(start_seconds * SAMPLE_RATE)
        end = round(end_seconds * SAMPLE_RATE)

        if start >= sample_count or end <= 0:
            dropped += 1
            continue

        if start < 0 or end > sample_count:
            clipped += 1

        labels[max(start, 0):min(end, sample_count)] = 1

    return labels, clipped, dropped


def read_sensor(
    archive,
    info,
):
    raw = archive.read(info)

    if len(raw) % (CHANNELS * 4) != 0:
        raise ValueError(
            f"{info.filename}: size {len(raw)} is not a "
            f"multiple of {CHANNELS * 4} bytes."
        )

    return (
        np.frombuffer(
            raw,
            dtype="<f4",
        )
        .reshape(-1, CHANNELS)
        .astype(np.float32)
    )


def create_window_rows(
    recording_id,
    split_name,
    processed_path,
    labels,
):
    rows = []

    starts = range(
        0,
        len(labels) - WINDOW_SAMPLES + 1,
        STRIDE_SAMPLES,
    )

    for window_index, start in enumerate(starts):
        end = start + WINDOW_SAMPLES
        eating = int(labels[start:end].sum())

        rows.append(
            {
                "split": split_name,
                "recording_id": recording_id,
                "file": processed_path.relative_to(ROOT).as_posix(),
                "window_index": window_index,
                "start_sample": start,
                "end_sample": end,
                "start_seconds": start / SAMPLE_RATE,
                "end_seconds": end / SAMPLE_RATE,
                "other_samples": WINDOW_SAMPLES - eating,
                "eating_samples": eating,
                "contains_eating": int(eating > 0),
            }
        )

    return rows


def main():
    print()
    print("CLEMSON ALL-DAY PREPROCESSING")
    print("=" * 60)
    print(f"Sample rate: {SAMPLE_RATE} Hz (kept as recorded)")
    print(f"Window:      {WINDOW_SECONDS} s ({WINDOW_SAMPLES} samples)")
    print(f"Stride:      {STRIDE_SECONDS} s ({STRIDE_SAMPLES} samples)")
    print()

    if not SENSOR_ARCHIVE.exists():
        raise FileNotFoundError(
            f"{SENSOR_ARCHIVE.relative_to(ROOT)} not found.\n"
            "Download it first with:\n"
            "python scripts/data/download_data.py "
            "--dataset clemson_allday --file transformed"
        )

    split, recording_to_split = load_split()
    events = load_events()

    window_rows = []
    recording_summaries = []
    seen = set()

    with zipfile.ZipFile(
        SENSOR_ARCHIVE,
        "r",
    ) as archive:
        members = sorted(
            (
                info
                for info in archive.infolist()
                if not info.is_dir()
                and info.filename.lower().endswith(".shm")
            ),
            key=lambda info: info.filename,
        )

        for position, info in enumerate(members, start=1):
            recording_id = normalize_id(info.filename)

            if recording_id not in recording_to_split:
                raise ValueError(
                    f"{info.filename}: recording not in split file."
                )

            if recording_id in seen:
                raise ValueError(
                    f"{recording_id}: more than one .shm file."
                )

            seen.add(recording_id)
            split_name = recording_to_split[recording_id]

            sensor = read_sensor(archive, info)
            recording_events = events.get(recording_id, [])

            labels, clipped, dropped = build_labels(
                len(sensor),
                recording_events,
            )

            split_dir = RECORDING_OUTPUT_DIR / split_name
            split_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = split_dir / f"{recording_id}.npz"

            np.savez(
                output_path,
                X=sensor,
                y=labels,
                recording_id=np.array(recording_id),
                sample_rate_hz=np.int16(SAMPLE_RATE),
            )

            rows = create_window_rows(
                recording_id,
                split_name,
                output_path,
                labels,
            )

            window_rows.extend(rows)

            eating_samples = int(labels.sum())

            recording_summaries.append(
                {
                    "recording_id": recording_id,
                    "split": split_name,
                    "samples": len(sensor),
                    "hours": len(sensor) / SAMPLE_RATE / 3600,
                    "events": len(recording_events),
                    "events_clipped": clipped,
                    "events_dropped": dropped,
                    "eating_samples": eating_samples,
                    "nonfinite_values": int(
                        (~np.isfinite(sensor)).sum()
                    ),
                    "windows": len(rows),
                }
            )

            print(
                f"[{position:03d}/{len(members):03d}] "
                f"{recording_id} [{split_name:<10}] "
                f"{len(sensor):,} samples, "
                f"{len(recording_events)} events, "
                f"{len(rows):,} windows"
            )

    missing = sorted(set(recording_to_split) - seen)

    if missing:
        raise ValueError(
            f"Recordings in split but not in archive: {missing}"
        )

    WINDOWS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with WINDOWS_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=WINDOW_FIELDS,
        )
        writer.writeheader()
        writer.writerows(window_rows)

    summary = {
        "dataset": "Clemson All-day",
        "sample_rate_hz": SAMPLE_RATE,
        "resampling": "none",
        "channels": CHANNELS,
        "labels": {
            "0": "not eating",
            "1": "inside a logged eating event",
        },
        "label_rule": (
            "sample i is 1 if round(start*15) <= i < round(end*15) "
            "for any event; events are clipped to the sensor length"
        ),
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "overlap": 1 - STRIDE_SECONDS / WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "recordings": len(recording_summaries),
        "total_windows": len(window_rows),
        "events_clipped": sum(
            row["events_clipped"] for row in recording_summaries
        ),
        "events_dropped": sum(
            row["events_dropped"] for row in recording_summaries
        ),
        "recordings_with_nonfinite_values": [
            row["recording_id"]
            for row in recording_summaries
            if row["nonfinite_values"]
        ],
        "recordings_shorter_than_one_window": [
            row["recording_id"]
            for row in recording_summaries
            if row["windows"] == 0
        ],
        "splits": {},
        "per_recording": recording_summaries,
    }

    for split_name in ("train", "validation", "test"):
        recordings = [
            row
            for row in recording_summaries
            if row["split"] == split_name
        ]
        windows = [
            row
            for row in window_rows
            if row["split"] == split_name
        ]
        samples = sum(row["samples"] for row in recordings)
        eating = sum(row["eating_samples"] for row in recordings)

        summary["splits"][split_name] = {
            "recordings": len(split[split_name]),
            "hours": samples / SAMPLE_RATE / 3600,
            "samples": samples,
            "eating_samples": eating,
            "eating_percent": eating / samples * 100 if samples else 0,
            "windows": len(windows),
            "windows_with_eating": sum(
                row["contains_eating"] for row in windows
            ),
        }

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
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Recordings:     {summary['recordings']}")
    print(f"Total windows:  {summary['total_windows']:,}")
    print(f"Events clipped: {summary['events_clipped']}")
    print(f"Events dropped: {summary['events_dropped']}")
    print()

    for split_name in ("train", "validation", "test"):
        details = summary["splits"][split_name]
        print(split_name.upper())
        print(f"  recordings:          {details['recordings']}")
        print(f"  hours:               {details['hours']:,.1f}")
        print(f"  eating:              {details['eating_percent']:.2f}%")
        print(f"  windows:             {details['windows']:,}")
        print(f"  windows with eating: {details['windows_with_eating']:,}")

    if summary["recordings_with_nonfinite_values"]:
        print()
        print(
            "WARNING: NaN/Inf found in "
            f"{summary['recordings_with_nonfinite_values']}"
        )

    print()
    print(f"Saved recordings under: {RECORDING_OUTPUT_DIR.relative_to(ROOT)}")
    print(f"Saved window index:     {WINDOWS_OUTPUT.relative_to(ROOT)}")
    print(f"Saved audit summary:    {AUDIT_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
