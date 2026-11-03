import csv
import json
import re
import statistics
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SENSOR_ARCHIVE = (
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

EVENTS_CSV = (
    ROOT
    / "data"
    / "processed"
    / "clemson_allday"
    / "events.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "data_audit"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "clemson_alignment_summary.json"
)

RECORDING_OUTPUT = (
    OUTPUT_DIR
    / "clemson_recording_alignment.csv"
)

EVENT_OUTPUT = (
    OUTPUT_DIR
    / "clemson_event_alignment.csv"
)

SAMPLE_RATE_HZ = 15
CHANNEL_COUNT = 6
BYTES_PER_VALUE = 4
BYTES_PER_SAMPLE = (
    CHANNEL_COUNT
    * BYTES_PER_VALUE
)

ID_PATTERN = re.compile(
    r"\bP\d{4}\b",
    re.IGNORECASE,
)


def normalize_id(value):
    match = ID_PATTERN.search(
        str(value)
    )

    if match is None:
        return None

    return match.group(
        0
    ).upper()


def load_recordings():
    recordings = {}

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
                row[
                    "recording_id"
                ]
            )

            recordings[
                recording_id
            ] = {
                **row,
                "recording_duration_seconds":
                    float(
                        row[
                            "recording_duration_seconds"
                        ]
                    ),
            }

    return recordings


def load_events():
    events = []

    with EVENTS_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:
            events.append(
                {
                    **row,

                    "event_number":
                        int(
                            row[
                                "event_number"
                            ]
                        ),

                    "start_offset_seconds":
                        float(
                            row[
                                "start_offset_seconds"
                            ]
                        ),

                    "end_offset_seconds":
                        float(
                            row[
                                "end_offset_seconds"
                            ]
                        ),

                    "duration_seconds":
                        float(
                            row[
                                "duration_seconds"
                            ]
                        ),
                }
            )

    return events


def load_sensor_files():
    sensors = {}

    with zipfile.ZipFile(
        SENSOR_ARCHIVE,
        "r",
    ) as archive:

        for info in archive.infolist():

            if (
                info.is_dir()
                or not info.filename
                .lower()
                .endswith(".shm")
            ):
                continue

            recording_id = (
                normalize_id(
                    info.filename
                )
            )

            if recording_id is None:
                continue

            if (
                info.file_size
                % BYTES_PER_SAMPLE
                != 0
            ):
                raise ValueError(
                    f"{info.filename} size "
                    f"{info.file_size} is not "
                    f"divisible by "
                    f"{BYTES_PER_SAMPLE}"
                )

            sample_count = (
                info.file_size
                // BYTES_PER_SAMPLE
            )

            duration_seconds = (
                sample_count
                / SAMPLE_RATE_HZ
            )

            sensors[
                recording_id
            ] = {
                "sensor_file":
                    info.filename,

                "sensor_bytes":
                    info.file_size,

                "sample_count":
                    sample_count,

                "sensor_duration_seconds":
                    duration_seconds,
            }

    return sensors


def group_events(events):
    grouped = {}

    for event in events:
        recording_id = (
            event[
                "recording_id"
            ]
        )

        grouped.setdefault(
            recording_id,
            [],
        ).append(
            event
        )

    return grouped


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[
                0
            ].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main():
    recordings = (
        load_recordings()
    )

    events = (
        load_events()
    )

    sensors = (
        load_sensor_files()
    )

    events_by_recording = (
        group_events(
            events
        )
    )

    recording_rows = []
    event_rows = []

    duration_differences = []

    recordings_with_short_sensor = []
    events_outside_sensor = []

    for recording_id in sorted(
        recordings
    ):
        recording = recordings[
            recording_id
        ]

        sensor = sensors.get(
            recording_id
        )

        if sensor is None:
            raise ValueError(
                f"No sensor file for "
                f"{recording_id}"
            )

        recording_events = (
            events_by_recording.get(
                recording_id,
                [],
            )
        )

        log_duration = (
            recording[
                "recording_duration_seconds"
            ]
        )

        sensor_duration = (
            sensor[
                "sensor_duration_seconds"
            ]
        )

        duration_difference = (
            sensor_duration
            - log_duration
        )

        duration_differences.append(
            duration_difference
        )

        sensor_shorter_than_log = (
            sensor_duration
            < log_duration
        )

        if sensor_shorter_than_log:
            recordings_with_short_sensor.append(
                recording_id
            )

        max_event_end = (
            max(
                (
                    event[
                        "end_offset_seconds"
                    ]
                    for event
                    in recording_events
                ),
                default=0,
            )
        )

        all_events_inside_sensor = True

        for event in recording_events:
            start_offset = (
                event[
                    "start_offset_seconds"
                ]
            )

            end_offset = (
                event[
                    "end_offset_seconds"
                ]
            )

            start_sample = int(
                start_offset
                * SAMPLE_RATE_HZ
            )

            end_sample = int(
                end_offset
                * SAMPLE_RATE_HZ
            )

            starts_inside = (
                0
                <= start_sample
                < sensor[
                    "sample_count"
                ]
            )

            ends_inside = (
                0
                < end_sample
                <= sensor[
                    "sample_count"
                ]
            )

            valid_interval = (
                start_sample
                < end_sample
            )

            aligned = (
                starts_inside
                and ends_inside
                and valid_interval
            )

            if not aligned:
                all_events_inside_sensor = (
                    False
                )

                events_outside_sensor.append(
                    {
                        "recording_id":
                            recording_id,

                        "event_number":
                            event[
                                "event_number"
                            ],

                        "event_type":
                            event[
                                "event_type"
                            ],

                        "start_offset_seconds":
                            start_offset,

                        "end_offset_seconds":
                            end_offset,

                        "sensor_duration_seconds":
                            sensor_duration,
                    }
                )

            event_rows.append(
                {
                    "recording_id":
                        recording_id,

                    "event_number":
                        event[
                            "event_number"
                        ],

                    "event_type":
                        event[
                            "event_type"
                        ],

                    "start_offset_seconds":
                        start_offset,

                    "end_offset_seconds":
                        end_offset,

                    "start_sample":
                        start_sample,

                    "end_sample":
                        end_sample,

                    "sensor_samples":
                        sensor[
                            "sample_count"
                        ],

                    "aligned":
                        aligned,
                }
            )

        recording_rows.append(
            {
                "recording_id":
                    recording_id,

                "sensor_file":
                    sensor[
                        "sensor_file"
                    ],

                "sample_count":
                    sensor[
                        "sample_count"
                    ],

                "sensor_duration_seconds":
                    round(
                        sensor_duration,
                        6,
                    ),

                "log_duration_seconds":
                    round(
                        log_duration,
                        6,
                    ),

                "sensor_minus_log_seconds":
                    round(
                        duration_difference,
                        6,
                    ),

                "event_count":
                    len(
                        recording_events
                    ),

                "max_event_end_seconds":
                    max_event_end,

                "sensor_shorter_than_log":
                    sensor_shorter_than_log,

                "all_events_inside_sensor":
                    all_events_inside_sensor,
            }
        )

    total_sensor_samples = sum(
        sensor[
            "sample_count"
        ]
        for sensor in sensors.values()
    )

    total_sensor_hours = (
        total_sensor_samples
        / SAMPLE_RATE_HZ
        / 3600
    )

    aligned_event_count = sum(
        row[
            "aligned"
        ]
        for row in event_rows
    )

    summary = {
        "dataset":
            "Clemson All-day Dataset",

        "sample_rate_hz":
            SAMPLE_RATE_HZ,

        "channels": [
            "linear_acc_x",
            "linear_acc_y",
            "linear_acc_z",
            "gyro_yaw",
            "gyro_pitch",
            "gyro_roll",
        ],

        "binary_layout": {
            "values_per_sample":
                CHANNEL_COUNT,

            "value_type":
                "float32",

            "bytes_per_value":
                BYTES_PER_VALUE,

            "bytes_per_sample":
                BYTES_PER_SAMPLE,
        },

        "recordings":
            len(
                recordings
            ),

        "sensor_recordings":
            len(
                sensors
            ),

        "events":
            len(
                events
            ),

        "aligned_events":
            aligned_event_count,

        "unaligned_events":
            len(
                events_outside_sensor
            ),

        "recordings_with_sensor_shorter_than_log":
            recordings_with_short_sensor,

        "total_sensor_samples":
            total_sensor_samples,

        "total_sensor_hours":
            total_sensor_hours,

        "duration_difference_seconds": {
            "minimum":
                min(
                    duration_differences
                ),

            "maximum":
                max(
                    duration_differences
                ),

            "mean":
                statistics.mean(
                    duration_differences
                ),

            "median":
                statistics.median(
                    duration_differences
                ),
        },

        "events_outside_sensor":
            events_outside_sensor,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    write_csv(
        RECORDING_OUTPUT,
        recording_rows,
    )

    write_csv(
        EVENT_OUTPUT,
        event_rows,
    )

    print()
    print(
        "CLEMSON SENSOR/EVENT ALIGNMENT"
    )
    print("=" * 60)

    print(
        f"Recordings:          "
        f"{summary['recordings']}"
    )

    print(
        f"Sensor recordings:   "
        f"{summary['sensor_recordings']}"
    )

    print(
        f"Events:              "
        f"{summary['events']}"
    )

    print(
        f"Aligned events:      "
        f"{summary['aligned_events']}"
    )

    print(
        f"Unaligned events:    "
        f"{summary['unaligned_events']}"
    )

    print()

    print(
        f"Sensor hours:        "
        f"{summary['total_sensor_hours']:.2f}"
    )

    print()

    print(
        "Sensor - log duration:"
    )

    differences = (
        summary[
            "duration_difference_seconds"
        ]
    )

    print(
        f"  min:    "
        f"{differences['minimum']:.2f} s"
    )

    print(
        f"  median: "
        f"{differences['median']:.2f} s"
    )

    print(
        f"  mean:   "
        f"{differences['mean']:.2f} s"
    )

    print(
        f"  max:    "
        f"{differences['maximum']:.2f} s"
    )

    print()

    print(
        "Recordings with sensor "
        "shorter than log:"
    )

    print(
        f"  "
        f"{len(recordings_with_short_sensor)}"
    )

    print()

    if events_outside_sensor:
        print(
            "EVENTS OUTSIDE SENSOR DATA"
        )
        print("-" * 60)

        for event in (
            events_outside_sensor[
                :20
            ]
        ):
            print(
                event
            )

        print()

    print(
        "Saved:"
    )

    print(
        f"  "
        f"{SUMMARY_OUTPUT.relative_to(ROOT)}"
    )

    print(
        f"  "
        f"{RECORDING_OUTPUT.relative_to(ROOT)}"
    )

    print(
        f"  "
        f"{EVENT_OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()