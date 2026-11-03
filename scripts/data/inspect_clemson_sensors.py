import csv
import json
import re
import struct
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "clemson_allday"
    / "SHMfiles.zip"
)

RECORDINGS_CSV = (
    ROOT
    / "data"
    / "processed"
    / "clemson_allday"
    / "recordings.csv"
)

OUTPUT = (
    ROOT
    / "results"
    / "data_audit"
    / "clemson_sensor_archive.json"
)

PARTICIPANT_PATTERN = re.compile(
    r"\bP\d{4}\b",
    re.IGNORECASE,
)

SAMPLE_RATE_HZ = 15
CHANNEL_COUNT = 6

CANDIDATE_LAYOUTS = {
    "float32_6ch": CHANNEL_COUNT * 4,
    "float64_6ch": CHANNEL_COUNT * 8,
    "int16_6ch": CHANNEL_COUNT * 2,
}


def format_size(size_bytes):
    size = float(size_bytes)

    for unit in (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ):
        if (
            size < 1024
            or unit == "TB"
        ):
            return (
                f"{size:.2f} {unit}"
            )

        size /= 1024

    return f"{size_bytes} B"


def normalize_id(value):
    match = PARTICIPANT_PATTERN.search(
        str(value)
    )

    if match is None:
        return None

    return (
        match.group(0)
        .upper()
    )


def load_event_recording_ids():
    if not RECORDINGS_CSV.exists():
        raise FileNotFoundError(
            RECORDINGS_CSV
        )

    ids = set()

    with RECORDINGS_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:
            recording_id = (
                row.get(
                    "recording_id"
                )
            )

            if recording_id:
                ids.add(
                    recording_id
                )

    return ids


def inspect_float32_prefix(
    archive,
    info,
):
    """
    Try interpreting the beginning of a file as
    little-endian float32 values.

    This is only diagnostic. It does not yet prove
    the final SHM format.
    """

    bytes_needed = (
        CHANNEL_COUNT
        * 4
        * 5
    )

    with archive.open(
        info,
        "r",
    ) as file:
        raw = file.read(
            bytes_needed
        )

    if (
        len(raw)
        < CHANNEL_COUNT * 4
    ):
        return {
            "bytes_read":
                len(raw),

            "rows":
                [],
        }

    value_count = (
        len(raw)
        // 4
    )

    raw = raw[
        : value_count * 4
    ]

    try:
        values = struct.unpack(
            f"<{value_count}f",
            raw,
        )

    except struct.error:
        return {
            "bytes_read":
                len(raw),

            "rows":
                [],
        }

    rows = []

    for index in range(
        0,
        len(values),
        CHANNEL_COUNT,
    ):
        row = values[
            index:
            index
            + CHANNEL_COUNT
        ]

        if (
            len(row)
            == CHANNEL_COUNT
        ):
            rows.append(
                [
                    float(value)
                    for value in row
                ]
            )

    return {
        "bytes_read":
            len(raw),

        "rows":
            rows[:5],
    }


def main():
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            ARCHIVE
        )

    event_recording_ids = (
        load_event_recording_ids()
    )

    with zipfile.ZipFile(
        ARCHIVE,
        "r",
    ) as archive:

        members = [
            info
            for info
            in archive.infolist()
            if not info.is_dir()
        ]

        shm_members = [
            info
            for info in members
            if info.filename
            .lower()
            .endswith(".shm")
        ]

        other_members = [
            info.filename
            for info in members
            if not info.filename
            .lower()
            .endswith(".shm")
        ]

        sensor_ids = []

        file_rows = []

        for info in shm_members:
            recording_id = (
                normalize_id(
                    info.filename
                )
            )

            if recording_id:
                sensor_ids.append(
                    recording_id
                )

            layout_results = {}

            for (
                layout_name,
                bytes_per_sample,
            ) in (
                CANDIDATE_LAYOUTS.items()
            ):
                divisible = (
                    info.file_size
                    % bytes_per_sample
                    == 0
                )

                if divisible:
                    samples = (
                        info.file_size
                        // bytes_per_sample
                    )

                    duration_hours = (
                        samples
                        / SAMPLE_RATE_HZ
                        / 3600
                    )

                else:
                    samples = None
                    duration_hours = None

                layout_results[
                    layout_name
                ] = {
                    "bytes_per_sample":
                        bytes_per_sample,

                    "exactly_divisible":
                        divisible,

                    "samples":
                        samples,

                    "duration_hours":
                        duration_hours,
                }

            file_rows.append(
                {
                    "filename":
                        info.filename,

                    "recording_id":
                        recording_id,

                    "compressed_size_bytes":
                        info.compress_size,

                    "uncompressed_size_bytes":
                        info.file_size,

                    "layouts":
                        layout_results,
                }
            )

        sensor_id_set = set(
            sensor_ids
        )

        duplicate_ids = sorted(
            {
                recording_id
                for recording_id
                in sensor_ids
                if sensor_ids.count(
                    recording_id
                ) > 1
            }
        )

        total_compressed = sum(
            info.compress_size
            for info in shm_members
        )

        total_uncompressed = sum(
            info.file_size
            for info in shm_members
        )

        layout_summary = {}

        for (
            layout_name,
            bytes_per_sample,
        ) in (
            CANDIDATE_LAYOUTS.items()
        ):
            divisible_files = [
                row
                for row in file_rows
                if row[
                    "layouts"
                ][
                    layout_name
                ][
                    "exactly_divisible"
                ]
            ]

            if (
                len(divisible_files)
                == len(file_rows)
                and file_rows
            ):
                total_samples = sum(
                    row[
                        "layouts"
                    ][
                        layout_name
                    ][
                        "samples"
                    ]
                    for row
                    in divisible_files
                )

                total_hours = (
                    total_samples
                    / SAMPLE_RATE_HZ
                    / 3600
                )

            else:
                total_samples = None
                total_hours = None

            layout_summary[
                layout_name
            ] = {
                "bytes_per_sample":
                    bytes_per_sample,

                "files_exactly_divisible":
                    len(
                        divisible_files
                    ),

                "total_files":
                    len(
                        file_rows
                    ),

                "total_samples":
                    total_samples,

                "total_hours_at_15hz":
                    total_hours,
            }

        prefix_samples = []

        for info in shm_members[
            :3
        ]:
            prefix_samples.append(
                {
                    "filename":
                        info.filename,

                    "float32_little_endian_preview":
                        inspect_float32_prefix(
                            archive,
                            info,
                        ),
                }
            )

    missing_sensor_for_event = sorted(
        event_recording_ids
        - sensor_id_set
    )

    sensors_without_event = sorted(
        sensor_id_set
        - event_recording_ids
    )

    summary = {
        "dataset":
            "Clemson All-day Dataset",

        "archive":
            ARCHIVE.name,

        "archive_size_bytes":
            ARCHIVE.stat().st_size,

        "files_total":
            len(
                members
            ),

        "shm_files":
            len(
                shm_members
            ),

        "other_files":
            other_members,

        "unique_sensor_ids":
            len(
                sensor_id_set
            ),

        "event_recording_ids":
            len(
                event_recording_ids
            ),

        "duplicate_sensor_ids":
            duplicate_ids,

        "event_recordings_without_sensor":
            missing_sensor_for_event,

        "sensor_recordings_without_event":
            sensors_without_event,

        "total_compressed_shm_bytes":
            total_compressed,

        "total_uncompressed_shm_bytes":
            total_uncompressed,

        "candidate_layouts":
            layout_summary,

        "float32_prefix_samples":
            prefix_samples,

        "first_20_files":
            file_rows[:20],
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
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
        "CLEMSON SENSOR ARCHIVE AUDIT"
    )
    print("=" * 60)

    print(
        f"Archive size:          "
        f"{format_size(summary['archive_size_bytes'])}"
    )

    print(
        f"Files in ZIP:          "
        f"{summary['files_total']}"
    )

    print(
        f".shm files:            "
        f"{summary['shm_files']}"
    )

    print(
        f"Unique sensor IDs:     "
        f"{summary['unique_sensor_ids']}"
    )

    print(
        f"Event recording IDs:   "
        f"{summary['event_recording_ids']}"
    )

    print()

    print(
        "Events without sensor: "
        f"{summary['event_recordings_without_sensor']}"
    )

    print(
        "Sensors without event: "
        f"{summary['sensor_recordings_without_event']}"
    )

    print(
        "Duplicate sensor IDs:  "
        f"{summary['duplicate_sensor_ids']}"
    )

    print()

    print(
        f"Compressed SHM size:   "
        f"{format_size(total_compressed)}"
    )

    print(
        f"Uncompressed SHM size: "
        f"{format_size(total_uncompressed)}"
    )

    print()
    print(
        "CANDIDATE BINARY LAYOUTS"
    )
    print("-" * 60)

    for (
        name,
        details,
    ) in layout_summary.items():

        print(
            f"{name}:"
        )

        print(
            "  divisible files: "
            f"{details['files_exactly_divisible']}"
            f"/{details['total_files']}"
        )

        if (
            details[
                "total_hours_at_15hz"
            ]
            is not None
        ):
            print(
                "  total duration:  "
                f"{details['total_hours_at_15hz']:.2f} h"
            )

    print()
    print(
        "FIRST 10 SHM FILES"
    )
    print("-" * 60)

    for row in file_rows[
        :10
    ]:
        print(
            f"{row['filename']:<35} "
            f"{format_size(row['uncompressed_size_bytes'])}"
        )

    print()
    print(
        "FLOAT32 PREVIEW"
    )
    print("-" * 60)

    for sample in prefix_samples:
        print()
        print(
            sample[
                "filename"
            ]
        )

        for row in sample[
            "float32_little_endian_preview"
        ][
            "rows"
        ]:
            print(
                "  "
                + " ".join(
                    f"{value: .5f}"
                    for value in row
                )
            )

    print()
    print(
        "Saved:"
    )

    print(
        f"  {OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()