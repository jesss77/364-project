import csv
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, time, timedelta
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "clemson_allday"
PROCESSED_DIR = ROOT / "data" / "processed" / "clemson_allday"
RESULTS_DIR = ROOT / "results" / "data_audit"

EVENT_ARCHIVE = RAW_DIR / "EVENTfiles.zip"
DEMOGRAPHICS_FILE = RAW_DIR / "ShimmerDemographics.xlsx"

EVENTS_OUTPUT = PROCESSED_DIR / "events.csv"
RECORDINGS_OUTPUT = PROCESSED_DIR / "recordings.csv"
VALIDATION_OUTPUT = RESULTS_DIR / "clemson_events_validation.json"

ID_PATTERN = re.compile(
    r"\bP\d{4}\b",
    re.IGNORECASE,
)


def decode_text(raw):
    for encoding in (
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ):
        try:
            return raw.decode(
                encoding
            )

        except UnicodeDecodeError:
            continue

    return raw.decode(
        "latin-1",
        errors="replace",
    )


def split_tokens(line):
    return re.split(
        r"\s+",
        line.strip(),
    )


def marker_type(line):
    tokens = split_tokens(
        line
    )

    if not tokens:
        return None

    marker = tokens[
        0
    ].upper()

    if marker in {
        "START",
        "END",
    }:
        return marker

    return None


def normalize_id(value):
    if value is None:
        return None

    match = ID_PATTERN.search(
        str(value)
    )

    if match is None:
        return None

    return match.group(
        0
    ).upper()


def load_demographic_ids():
    workbook = load_workbook(
        DEMOGRAPHICS_FILE,
        read_only=True,
        data_only=True,
    )

    participant_ids = set()

    for worksheet in workbook.worksheets:
        rows = worksheet.iter_rows(
            values_only=True
        )

        try:
            header = next(
                rows
            )

        except StopIteration:
            continue

        participant_column = None

        for index, value in enumerate(
            header
        ):
            if (
                str(value)
                .strip()
                .lower()
                == "participant id"
            ):
                participant_column = index
                break

        if participant_column is None:
            continue

        for row in rows:
            if (
                participant_column
                >= len(row)
            ):
                continue

            participant_id = normalize_id(
                row[
                    participant_column
                ]
            )

            if participant_id:
                participant_ids.add(
                    participant_id
                )

    workbook.close()

    return participant_ids


def parse_hms(value):
    parts = value.strip().split(
        ":"
    )

    if len(parts) != 3:
        raise ValueError(
            f"Invalid time value: {value}"
        )

    try:
        hour = int(
            parts[0]
        )

        minute = int(
            parts[1]
        )

        second = int(
            parts[2]
        )

    except ValueError as error:
        raise ValueError(
            f"Invalid time value: {value}"
        ) from error

    if hour < 0:
        raise ValueError(
            f"Invalid hour: {hour}"
        )

    if not 0 <= minute <= 59:
        raise ValueError(
            f"Invalid minute: {minute}"
        )

    if not 0 <= second <= 59:
        raise ValueError(
            f"Invalid second: {second}"
        )

    day_offset, normalized_hour = divmod(
        hour,
        24,
    )

    return (
        day_offset,
        time(
            normalized_hour,
            minute,
            second,
        ),
    )


def datetime_from_date_and_clock(
    base_date,
    clock_value,
):
    day_offset, clock_time = (
        parse_hms(
            clock_value
        )
    )

    result = datetime.combine(
        base_date,
        clock_time,
    )

    return result + timedelta(
        days=day_offset
    )


def parse_marker(
    line,
    expected_marker,
):
    tokens = split_tokens(
        line
    )

    if len(tokens) < 3:
        raise ValueError(
            f"Malformed marker line: {line}"
        )

    actual_marker = tokens[
        0
    ].upper()

    if (
        actual_marker
        != expected_marker
    ):
        raise ValueError(
            f"Expected {expected_marker}, "
            f"found {actual_marker}: {line}"
        )

    try:
        marker_date = datetime.strptime(
            tokens[1],
            "%Y-%m-%d",
        ).date()

    except ValueError as error:
        raise ValueError(
            f"Invalid date in marker: {line}"
        ) from error

    return datetime_from_date_and_clock(
        marker_date,
        tokens[2],
    )


def event_datetime(
    recording_start,
    clock_value,
    minimum=None,
):
    result = datetime_from_date_and_clock(
        recording_start.date(),
        clock_value,
    )

    threshold = (
        minimum
        if minimum is not None
        else recording_start
    )

    while result < threshold:
        result += timedelta(
            days=1
        )

    return result


def parse_event_line(
    line,
    recording_start,
):
    tokens = split_tokens(
        line
    )

    if len(tokens) < 3:
        raise ValueError(
            f"Malformed event line: {line}"
        )

    event_type = tokens[
        0
    ]

    event_start = event_datetime(
        recording_start,
        tokens[1],
    )

    event_end = event_datetime(
        recording_start,
        tokens[2],
        minimum=event_start,
    )

    metadata = tokens[
        3:
    ]

    return {
        "event_type":
            event_type,

        "event_start":
            event_start,

        "event_end":
            event_end,

        "duration_seconds":
            int(
                (
                    event_end
                    - event_start
                ).total_seconds()
            ),

        "location":
            (
                metadata[0]
                if len(metadata) > 0
                else ""
            ),

        "seconds_status":
            (
                metadata[1]
                if len(metadata) > 1
                else ""
            ),

        "social_context":
            (
                metadata[2]
                if len(metadata) > 2
                else ""
            ),

        "food":
            (
                metadata[3]
                if len(metadata) > 3
                else ""
            ),

        "activity":
            (
                " ".join(
                    metadata[4:]
                )
                if len(metadata) > 4
                else ""
            ),
    }


def count_overlaps(events):
    ordered = sorted(
        events,
        key=lambda event:
            event[
                "event_start"
            ],
    )

    overlaps = 0

    for previous, current in zip(
        ordered,
        ordered[1:],
    ):
        if (
            current[
                "event_start"
            ]
            < previous[
                "event_end"
            ]
        ):
            overlaps += 1

    return overlaps


def contains_overflow_hour(line):
    tokens = split_tokens(
        line
    )

    if marker_type(
        line
    ) is not None:
        time_tokens = tokens[
            2:3
        ]

    else:
        time_tokens = tokens[
            1:3
        ]

    for token in time_tokens:
        try:
            hour = int(
                token.split(
                    ":"
                )[0]
            )

            if hour >= 24:
                return True

        except (
            ValueError,
            IndexError,
        ):
            continue

    return False


def parse_archive():
    demographic_ids = (
        load_demographic_ids()
    )

    event_rows = []
    recording_rows = []

    missing_start_files = []
    missing_end_files = []

    malformed_markers = []
    malformed_events = []

    nonpositive_events = []
    out_of_bounds_events = []

    overlapping_recordings = {}

    overflow_timestamp_examples = []

    event_type_counts = Counter()
    seconds_status_counts = Counter()

    with zipfile.ZipFile(
        EVENT_ARCHIVE,
        "r",
    ) as archive:

        corrupt_member = (
            archive.testzip()
        )

        if corrupt_member is not None:
            raise ValueError(
                "Corrupt archive member: "
                f"{corrupt_member}"
            )

        members = sorted(
            (
                info
                for info in archive.infolist()
                if (
                    not info.is_dir()
                    and info.filename
                    .lower()
                    .endswith(".txt")
                )
            ),
            key=lambda info:
                info.filename,
        )

        for info in members:
            text = decode_text(
                archive.read(
                    info.filename
                )
            )

            lines = [
                line.strip()
                for line
                in text.splitlines()
                if line.strip()
            ]

            recording_id = normalize_id(
                info.filename
            )

            start_lines = [
                line
                for line in lines
                if marker_type(
                    line
                ) == "START"
            ]

            end_lines = [
                line
                for line in lines
                if marker_type(
                    line
                ) == "END"
            ]

            if not start_lines:
                missing_start_files.append(
                    info.filename
                )

                continue

            try:
                recording_start = (
                    parse_marker(
                        start_lines[0],
                        "START",
                    )
                )

            except ValueError as error:
                malformed_markers.append(
                    {
                        "source_file":
                            info.filename,

                        "line":
                            start_lines[0],

                        "error":
                            str(error),
                    }
                )

                continue

            recording_end = None

            if end_lines:
                try:
                    recording_end = (
                        parse_marker(
                            end_lines[-1],
                            "END",
                        )
                    )

                    while (
                        recording_end
                        < recording_start
                    ):
                        recording_end += timedelta(
                            days=1
                        )

                except ValueError as error:
                    malformed_markers.append(
                        {
                            "source_file":
                                info.filename,

                            "line":
                                end_lines[-1],

                            "error":
                                str(error),
                        }
                    )

            else:
                missing_end_files.append(
                    info.filename
                )

            for line in lines:
                if contains_overflow_hour(
                    line
                ):
                    overflow_timestamp_examples.append(
                        {
                            "source_file":
                                info.filename,

                            "line":
                                line,
                        }
                    )

            file_events = []

            for line_number, line in enumerate(
                lines,
                start=1,
            ):
                if marker_type(
                    line
                ) is not None:
                    continue

                try:
                    event = parse_event_line(
                        line,
                        recording_start,
                    )

                except ValueError as error:
                    malformed_events.append(
                        {
                            "source_file":
                                info.filename,

                            "line_number":
                                line_number,

                            "line":
                                line,

                            "error":
                                str(error),
                        }
                    )

                    continue

                if (
                    event[
                        "duration_seconds"
                    ]
                    <= 0
                ):
                    nonpositive_events.append(
                        {
                            "source_file":
                                info.filename,

                            "line":
                                line,
                        }
                    )

                if recording_end is not None:
                    if (
                        event[
                            "event_start"
                        ]
                        < recording_start
                        or event[
                            "event_end"
                        ]
                        > recording_end
                    ):
                        out_of_bounds_events.append(
                            {
                                "source_file":
                                    info.filename,

                                "event_type":
                                    event[
                                        "event_type"
                                    ],

                                "event_start":
                                    event[
                                        "event_start"
                                    ].isoformat(
                                        sep=" "
                                    ),

                                "event_end":
                                    event[
                                        "event_end"
                                    ].isoformat(
                                        sep=" "
                                    ),
                            }
                        )

                start_offset_seconds = int(
                    (
                        event[
                            "event_start"
                        ]
                        - recording_start
                    ).total_seconds()
                )

                end_offset_seconds = int(
                    (
                        event[
                            "event_end"
                        ]
                        - recording_start
                    ).total_seconds()
                )

                event_number = (
                    len(
                        file_events
                    )
                    + 1
                )

                event_rows.append(
                    {
                        "recording_id":
                            recording_id,

                        "event_number":
                            event_number,

                        "source_file":
                            info.filename,

                        "event_type":
                            event[
                                "event_type"
                            ],

                        "event_start":
                            event[
                                "event_start"
                            ].isoformat(
                                sep=" "
                            ),

                        "event_end":
                            event[
                                "event_end"
                            ].isoformat(
                                sep=" "
                            ),

                        "start_offset_seconds":
                            start_offset_seconds,

                        "end_offset_seconds":
                            end_offset_seconds,

                        "duration_seconds":
                            event[
                                "duration_seconds"
                            ],

                        "location":
                            event[
                                "location"
                            ],

                        "seconds_status":
                            event[
                                "seconds_status"
                            ],

                        "social_context":
                            event[
                                "social_context"
                            ],

                        "food":
                            event[
                                "food"
                            ],

                        "activity":
                            event[
                                "activity"
                            ],

                        "has_demographics":
                            (
                                recording_id
                                in demographic_ids
                            ),
                    }
                )

                file_events.append(
                    event
                )

                event_type_counts[
                    event[
                        "event_type"
                    ]
                ] += 1

                seconds_status_counts[
                    event[
                        "seconds_status"
                    ]
                    or "<blank>"
                ] += 1

            overlap_count = count_overlaps(
                file_events
            )

            if overlap_count:
                overlapping_recordings[
                    info.filename
                ] = overlap_count

            if recording_end is None:
                recording_duration_seconds = ""

            else:
                recording_duration_seconds = int(
                    (
                        recording_end
                        - recording_start
                    ).total_seconds()
                )

            recording_rows.append(
                {
                    "recording_id":
                        recording_id,

                    "source_file":
                        info.filename,

                    "recording_start":
                        recording_start
                        .isoformat(
                            sep=" "
                        ),

                    "recording_end":
                        (
                            recording_end
                            .isoformat(
                                sep=" "
                            )
                            if recording_end
                            else ""
                        ),

                    "recording_duration_seconds":
                        recording_duration_seconds,

                    "event_count":
                        len(
                            file_events
                        ),

                    "has_demographics":
                        (
                            recording_id
                            in demographic_ids
                        ),
                }
            )

    return {
        "events":
            event_rows,

        "recordings":
            recording_rows,

        "event_type_counts":
            event_type_counts,

        "seconds_status_counts":
            seconds_status_counts,

        "missing_start_files":
            missing_start_files,

        "missing_end_files":
            missing_end_files,

        "malformed_markers":
            malformed_markers,

        "malformed_events":
            malformed_events,

        "nonpositive_events":
            nonpositive_events,

        "out_of_bounds_events":
            out_of_bounds_events,

        "overlapping_recordings":
            overlapping_recordings,

        "overflow_timestamp_examples":
            overflow_timestamp_examples,

        "demographic_ids":
            demographic_ids,
    }


def write_csv(
    path,
    rows,
):
    if not rows:
        raise ValueError(
            f"No rows available for {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
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
    if not EVENT_ARCHIVE.exists():
        raise FileNotFoundError(
            EVENT_ARCHIVE
        )

    if not DEMOGRAPHICS_FILE.exists():
        raise FileNotFoundError(
            DEMOGRAPHICS_FILE
        )

    result = parse_archive()

    events = result[
        "events"
    ]

    recordings = result[
        "recordings"
    ]

    write_csv(
        EVENTS_OUTPUT,
        events,
    )

    write_csv(
        RECORDINGS_OUTPUT,
        recordings,
    )

    recording_ids = {
        row[
            "recording_id"
        ]
        for row in recordings
    }

    event_recordings = {
        row[
            "source_file"
        ]
        for row in events
    }

    demographic_ids = result[
        "demographic_ids"
    ]

    total_event_seconds = sum(
        row[
            "duration_seconds"
        ]
        for row in events
    )

    total_log_span_seconds = sum(
        row[
            "recording_duration_seconds"
        ]
        for row in recordings
        if isinstance(
            row[
                "recording_duration_seconds"
            ],
            int,
        )
    )

    validation = {
        "dataset":
            "Clemson All-day Dataset",

        "source_files_parsed":
            len(
                recordings
            ),

        "unique_recording_ids":
            len(
                recording_ids
            ),

        "recordings_with_events":
            len(
                event_recordings
            ),

        "total_event_rows":
            len(
                events
            ),

        "total_event_hours":
            total_event_seconds
            / 3600,

        "total_start_end_log_span_hours":
            total_log_span_seconds
            / 3600,

        "event_type_counts":
            dict(
                sorted(
                    result[
                        "event_type_counts"
                    ].items()
                )
            ),

        "seconds_status_counts":
            dict(
                sorted(
                    result[
                        "seconds_status_counts"
                    ].items()
                )
            ),

        "missing_start_files":
            result[
                "missing_start_files"
            ],

        "missing_end_files":
            result[
                "missing_end_files"
            ],

        "malformed_markers":
            result[
                "malformed_markers"
            ],

        "malformed_events":
            result[
                "malformed_events"
            ],

        "nonpositive_events":
            result[
                "nonpositive_events"
            ],

        "out_of_bounds_events":
            result[
                "out_of_bounds_events"
            ],

        "overlapping_recordings":
            result[
                "overlapping_recordings"
            ],

        "overflow_timestamp_count":
            len(
                result[
                    "overflow_timestamp_examples"
                ]
            ),

        "overflow_timestamp_examples":
            result[
                "overflow_timestamp_examples"
            ][:20],

        "recording_ids_missing_demographics":
            sorted(
                recording_ids
                - demographic_ids
            ),
    }

    VALIDATION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with VALIDATION_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            validation,
            file,
            indent=2,
        )

    print()
    print(
        "CLEMSON EVENT PREPROCESSING"
    )
    print("=" * 60)

    print(
        f"Source files parsed:    "
        f"{validation['source_files_parsed']}"
    )

    print(
        f"Recording IDs:          "
        f"{validation['unique_recording_ids']}"
    )

    print(
        f"Recordings w/events:    "
        f"{validation['recordings_with_events']}"
    )

    print(
        f"Event rows:             "
        f"{validation['total_event_rows']}"
    )

    print(
        f"Event hours:            "
        f"{validation['total_event_hours']:.2f}"
    )

    print(
        f"START-END log span:     "
        f"{validation['total_start_end_log_span_hours']:.2f} h"
    )

    print()

    print(
        f"Missing START:          "
        f"{len(validation['missing_start_files'])}"
    )

    print(
        f"Missing END:            "
        f"{len(validation['missing_end_files'])}"
    )

    print(
        f"Malformed markers:      "
        f"{len(validation['malformed_markers'])}"
    )

    print(
        f"Malformed events:       "
        f"{len(validation['malformed_events'])}"
    )

    print(
        f"Nonpositive events:     "
        f"{len(validation['nonpositive_events'])}"
    )

    print(
        f"Out-of-bounds events:   "
        f"{len(validation['out_of_bounds_events'])}"
    )

    print(
        f"Overlap recordings:     "
        f"{len(validation['overlapping_recordings'])}"
    )

    print(
        f"Overflow timestamps:    "
        f"{validation['overflow_timestamp_count']}"
    )

    print()

    print(
        "Missing demographics:  "
        f"{validation['recording_ids_missing_demographics']}"
    )

    print()

    print(
        "Note: START-END log span is not assumed "
        "to equal usable sensor duration."
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"  {EVENTS_OUTPUT.relative_to(ROOT)}"
    )

    print(
        f"  {RECORDINGS_OUTPUT.relative_to(ROOT)}"
    )

    print(
        f"  {VALIDATION_OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()