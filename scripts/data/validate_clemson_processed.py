import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from preprocess_clemson import (
    CHANNELS,
    RECORDING_OUTPUT_DIR,
    ROOT,
    SAMPLE_RATE,
    STRIDE_SAMPLES,
    WINDOW_SAMPLES,
    WINDOWS_OUTPUT,
    build_labels,
    load_events,
    load_split,
)


ALIGNMENT_PATH = (
    ROOT
    / "results"
    / "data_audit"
    / "clemson_recording_alignment.csv"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "data_audit"
    / "clemson_processed_validation.json"
)


def load_expected_lengths():
    with ALIGNMENT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return {
            row["recording_id"]: int(row["sample_count"])
            for row in csv.DictReader(file)
        }


def validate_recordings(
    recording_to_split,
    events,
    expected_lengths,
):
    files = sorted(RECORDING_OUTPUT_DIR.rglob("*.npz"))
    errors = []
    summary = defaultdict(
        lambda: {
            "files": 0,
            "samples": 0,
            "eating_samples": 0,
        }
    )
    label_cumsums = {}

    if len(files) != len(recording_to_split):
        errors.append(
            f"Expected {len(recording_to_split)} recording files, "
            f"found {len(files)}."
        )

    found = set()

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            X = data["X"]
            y = data["y"]
            recording_id = str(data["recording_id"])
            sample_rate = int(data["sample_rate_hz"])

        name = path.name
        split_name = recording_to_split.get(recording_id)

        if split_name is None:
            errors.append(f"{name}: unknown recording {recording_id}")
            continue

        if recording_id in found:
            errors.append(f"{name}: duplicate recording {recording_id}")

        found.add(recording_id)

        if path.parent != RECORDING_OUTPUT_DIR / split_name:
            errors.append(f"{name}: stored in wrong split folder")

        if path.stem != recording_id:
            errors.append(f"{name}: file name does not match {recording_id}")

        if X.ndim != 2 or X.shape[1] != CHANNELS:
            errors.append(f"{name}: unexpected X shape {X.shape}")

        if len(X) != len(y):
            errors.append(f"{name}: X/y length mismatch")

        if X.dtype != np.float32:
            errors.append(f"{name}: X dtype is {X.dtype}")

        if y.dtype != np.uint8:
            errors.append(f"{name}: y dtype is {y.dtype}")

        if sample_rate != SAMPLE_RATE:
            errors.append(f"{name}: sample rate is {sample_rate}")

        expected_length = expected_lengths.get(recording_id)

        if expected_length != len(X):
            errors.append(
                f"{name}: {len(X)} samples, alignment audit "
                f"says {expected_length}"
            )

        if not np.isfinite(X).all():
            errors.append(f"{name}: X contains NaN/Inf")

        unexpected = set(np.unique(y).tolist()) - {0, 1}

        if unexpected:
            errors.append(f"{name}: unexpected labels {sorted(unexpected)}")

        expected_labels, _, _ = build_labels(
            len(y),
            events.get(recording_id, []),
        )

        if not np.array_equal(y, expected_labels):
            errors.append(f"{name}: labels do not match events.csv")

        if events.get(recording_id) and not y.any():
            errors.append(f"{name}: has events but no eating samples")

        summary[split_name]["files"] += 1
        summary[split_name]["samples"] += len(y)
        summary[split_name]["eating_samples"] += int(y.sum())

        label_cumsums[recording_id] = np.concatenate(
            ([0], np.cumsum(y, dtype=np.int64))
        )

    missing = sorted(set(recording_to_split) - found)

    if missing:
        errors.append(f"Recordings missing from processed data: {missing}")

    return summary, label_cumsums, errors


def validate_windows(
    recording_to_split,
    label_cumsums,
):
    errors = []
    counts = Counter()
    per_recording = Counter()
    previous = {}

    with WINDOWS_OUTPUT.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            recording_id = row["recording_id"]
            split_name = row["split"]
            start = int(row["start_sample"])
            end = int(row["end_sample"])
            where = f"windows.csv row {row_number}"

            if recording_to_split.get(recording_id) != split_name:
                errors.append(f"{where}: wrong split for {recording_id}")

            if end - start != WINDOW_SAMPLES:
                errors.append(f"{where}: window length {end - start}")

            if recording_id in previous:
                if start - previous[recording_id] != STRIDE_SAMPLES:
                    errors.append(
                        f"{where}: stride {start - previous[recording_id]}"
                    )
            elif start != 0:
                errors.append(f"{where}: first window starts at {start}")

            previous[recording_id] = start

            cumsum = label_cumsums.get(recording_id)

            if cumsum is not None:
                if end > len(cumsum) - 1:
                    errors.append(f"{where}: window runs past recording end")
                else:
                    eating = int(cumsum[end] - cumsum[start])

                    if eating != int(row["eating_samples"]):
                        errors.append(
                            f"{where}: eating_samples is "
                            f"{row['eating_samples']}, labels give {eating}"
                        )

                    if int(row["contains_eating"]) != int(eating > 0):
                        errors.append(f"{where}: contains_eating is wrong")

            counts[split_name] += 1
            per_recording[recording_id] += 1

    for recording_id, cumsum in label_cumsums.items():
        length = len(cumsum) - 1

        if length >= WINDOW_SAMPLES:
            expected = (length - WINDOW_SAMPLES) // STRIDE_SAMPLES + 1
        else:
            expected = 0

        if per_recording[recording_id] != expected:
            errors.append(
                f"{recording_id}: {per_recording[recording_id]} windows, "
                f"expected {expected}"
            )

    return counts, errors


def main():
    print()
    print("CLEMSON PROCESSED DATA VALIDATION")
    print("=" * 60)

    split, recording_to_split = load_split()
    events = load_events()
    expected_lengths = load_expected_lengths()

    if set(expected_lengths) != set(recording_to_split):
        raise ValueError(
            "Split file and alignment audit list different recordings."
        )

    sample_summary, label_cumsums, recording_errors = validate_recordings(
        recording_to_split,
        events,
        expected_lengths,
    )

    window_counts, window_errors = validate_windows(
        recording_to_split,
        label_cumsums,
    )

    errors = recording_errors + window_errors

    output = {
        "dataset": "Clemson All-day",
        "sample_rate_hz": SAMPLE_RATE,
        "channels": CHANNELS,
        "window_samples": WINDOW_SAMPLES,
        "stride_samples": STRIDE_SAMPLES,
        "splits": {},
        "errors": errors,
    }

    for split_name in ("train", "validation", "test"):
        data = sample_summary[split_name]
        samples = data["samples"]
        eating = data["eating_samples"]

        output["splits"][split_name] = {
            "recordings": len(split[split_name]),
            "files": data["files"],
            "samples": samples,
            "windows": window_counts[split_name],
            "eating_samples": eating,
            "eating_percent": eating / samples * 100 if samples else 0,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    for split_name in ("train", "validation", "test"):
        data = output["splits"][split_name]
        print()
        print(split_name.upper())
        print(f"  recordings: {data['recordings']}")
        print(f"  files:      {data['files']}")
        print(f"  windows:    {data['windows']:,}")
        print(f"  samples:    {data['samples']:,}")
        print(f"  eating:     {data['eating_percent']:.2f}%")

    print()
    print("=" * 60)

    if errors:
        print(f"VALIDATION FAILED: {len(errors)} issue(s)")

        for error in errors[:50]:
            print(f"  - {error}")

        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more in {OUTPUT_PATH.name}")

        raise SystemExit(1)

    print("VALIDATION PASSED")
    print("All recordings, labels, splits and windows are consistent.")
    print()
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
