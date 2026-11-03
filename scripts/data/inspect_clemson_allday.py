import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw" / "clemson_allday"
OUTPUT_DIR = ROOT / "results" / "data_audit"

EVENT_ARCHIVE = DATA_DIR / "EVENTfiles.zip"
DEMOGRAPHICS_FILE = DATA_DIR / "ShimmerDemographics.xlsx"

PARTICIPANT_PATTERN = re.compile(
    r"\bP?(\d{4})\b",
    re.IGNORECASE,
)


def normalize_participant_id(text):
    match = PARTICIPANT_PATTERN.search(
        str(text)
    )

    if match is None:
        return None

    return f"P{match.group(1)}"


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


def inspect_event_archive():
    if not EVENT_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Missing {EVENT_ARCHIVE}"
        )

    print(
        "Inspecting "
        f"{EVENT_ARCHIVE.relative_to(ROOT)}..."
    )

    with zipfile.ZipFile(
        EVENT_ARCHIVE,
        "r",
    ) as archive:

        corrupt_member = archive.testzip()

        if corrupt_member is not None:
            raise ValueError(
                "Corrupt archive member: "
                f"{corrupt_member}"
            )

        members = [
            info
            for info in archive.infolist()
            if not info.is_dir()
        ]

        extensions = Counter(
            Path(info.filename)
            .suffix.lower()
            or "<none>"
            for info in members
        )

        participant_ids = set()

        file_summaries = []

        total_nonempty_lines = 0

        for info in members:
            participant_id = (
                normalize_participant_id(
                    info.filename
                )
            )

            if participant_id:
                participant_ids.add(
                    participant_id
                )

            raw = archive.read(
                info.filename
            )

            text = decode_text(
                raw
            )

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            total_nonempty_lines += len(
                lines
            )

            preview = lines[:5]

            file_summaries.append(
                {
                    "filename":
                        info.filename,
                    "participant_id":
                        participant_id,
                    "size_bytes":
                        info.file_size,
                    "nonempty_lines":
                        len(lines),
                    "preview":
                        preview,
                }
            )

    summary = {
        "archive":
            EVENT_ARCHIVE.name,
        "archive_valid":
            True,
        "file_count":
            len(members),
        "extensions":
            dict(
                sorted(
                    extensions.items()
                )
            ),
        "participant_ids_found":
            sorted(
                participant_ids
            ),
        "participant_count":
            len(participant_ids),
        "total_nonempty_lines":
            total_nonempty_lines,
        "files":
            file_summaries,
    }

    return summary


def inspect_demographics():
    if not DEMOGRAPHICS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DEMOGRAPHICS_FILE}"
        )

    print(
        "Inspecting "
        f"{DEMOGRAPHICS_FILE.relative_to(ROOT)}..."
    )

    workbook = load_workbook(
        DEMOGRAPHICS_FILE,
        read_only=True,
        data_only=True,
    )

    sheets = []

    all_participant_ids = set()

    for worksheet in workbook.worksheets:
        rows = list(
            worksheet.iter_rows(
                values_only=True
            )
        )

        nonempty_rows = [
            row
            for row in rows
            if any(
                value is not None
                for value in row
            )
        ]

        preview = []

        for row in nonempty_rows[:5]:
            preview.append(
                [
                    value
                    for value in row
                ]
            )

        participant_ids = set()

        for row in nonempty_rows:
            for value in row:
                if value is None:
                    continue

                participant_id = (
                    normalize_participant_id(
                        value
                    )
                )

                if participant_id:
                    participant_ids.add(
                        participant_id
                    )

        all_participant_ids.update(
            participant_ids
        )

        sheets.append(
            {
                "name":
                    worksheet.title,
                "max_row":
                    worksheet.max_row,
                "max_column":
                    worksheet.max_column,
                "nonempty_rows":
                    len(nonempty_rows),
                "participant_count_detected":
                    len(participant_ids),
                "preview":
                    preview,
            }
        )

    workbook.close()

    return {
        "workbook":
            DEMOGRAPHICS_FILE.name,
        "sheet_count":
            len(sheets),
        "sheets":
            sheets,
        "participant_ids_found":
            sorted(
                all_participant_ids
            ),
        "participant_count":
            len(all_participant_ids),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_summary = (
        inspect_event_archive()
    )

    demographics_summary = (
        inspect_demographics()
    )

    event_ids = set(
        event_summary[
            "participant_ids_found"
        ]
    )

    demographic_ids = set(
        demographics_summary[
            "participant_ids_found"
        ]
    )

    audit = {
        "dataset":
            "Clemson All-day Dataset",
        "event_archive":
            event_summary,
        "demographics":
            demographics_summary,
        "participant_id_comparison": {
            "event_participants":
                len(event_ids),
            "demographic_participants":
                len(demographic_ids),
            "in_events_not_demographics":
                sorted(
                    event_ids
                    - demographic_ids
                ),
            "in_demographics_not_events":
                sorted(
                    demographic_ids
                    - event_ids
                ),
        },
    }

    output_path = (
        OUTPUT_DIR
        / "clemson_allday_structure.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
            default=str,
        )

    print()
    print(
        "CLEMSON ALL-DAY AUDIT"
    )
    print("=" * 60)

    print(
        "Ground-truth files:   "
        f"{event_summary['file_count']}"
    )

    print(
        "Event participant IDs:"
        f" {event_summary['participant_count']}"
    )

    print(
        "Event extensions:     "
        f"{event_summary['extensions']}"
    )

    print(
        "Total nonempty lines: "
        f"{event_summary['total_nonempty_lines']}"
    )

    print()

    print(
        "Demographic IDs:      "
        f"{demographics_summary['participant_count']}"
    )

    print(
        "Workbook sheets:      "
        f"{demographics_summary['sheet_count']}"
    )

    comparison = audit[
        "participant_id_comparison"
    ]

    print(
        "Events not demographics: "
        f"{comparison['in_events_not_demographics']}"
    )

    print(
        "Demographics not events: "
        f"{comparison['in_demographics_not_events']}"
    )

    print()
    print(
        "FIRST 5 GROUND-TRUTH FILES"
    )
    print("-" * 60)

    for item in event_summary[
        "files"
    ][:5]:
        print()
        print(
            f"FILE: {item['filename']}"
        )

        print(
            f"Participant: "
            f"{item['participant_id']}"
        )

        print(
            f"Lines: "
            f"{item['nonempty_lines']}"
        )

        for line in item[
            "preview"
        ]:
            print(
                f"  {line}"
            )

    print()
    print(
        "DEMOGRAPHICS PREVIEW"
    )
    print("-" * 60)

    for sheet in demographics_summary[
        "sheets"
    ]:
        print()
        print(
            f"Sheet: {sheet['name']}"
        )

        print(
            f"Rows: "
            f"{sheet['nonempty_rows']}"
        )

        for row in sheet[
            "preview"
        ]:
            print(
                f"  {row}"
            )

    print()
    print(
        "Saved: "
        f"{output_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()