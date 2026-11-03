import json
import re
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "clemson_allday"
    / "EVENTfiles.zip"
)

OUTPUT = (
    ROOT
    / "results"
    / "data_audit"
    / "clemson_event_marker_audit.json"
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


def first_token(line):
    parts = re.split(
        r"\s+",
        line.strip(),
    )

    if not parts:
        return "<empty>"

    return parts[0]


def main():
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            ARCHIVE
        )

    records = []

    with zipfile.ZipFile(
        ARCHIVE,
        "r",
    ) as archive:

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
            key=lambda item:
                item.filename,
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

            start_lines = [
                line
                for line in lines
                if first_token(
                    line
                ).upper()
                == "START"
            ]

            end_lines = [
                line
                for line in lines
                if first_token(
                    line
                ).upper()
                == "END"
            ]

            event_lines = [
                line
                for line in lines
                if (
                    first_token(
                        line
                    ).upper()
                    not in {
                        "START",
                        "END",
                    }
                )
            ]

            date_tokens = sorted(
                set(
                    re.findall(
                        r"\b\d{4}-\d{2}-\d{2}\b",
                        text,
                    )
                )
            )

            records.append(
                {
                    "filename":
                        info.filename,

                    "has_start":
                        bool(
                            start_lines
                        ),

                    "has_end":
                        bool(
                            end_lines
                        ),

                    "line_count":
                        len(
                            lines
                        ),

                    "event_line_count":
                        len(
                            event_lines
                        ),

                    "first_token":
                        (
                            first_token(
                                lines[0]
                            )
                            if lines
                            else "<empty>"
                        ),

                    "start_lines":
                        start_lines,

                    "end_lines":
                        end_lines,

                    "date_tokens":
                        date_tokens,

                    "lines":
                        lines,
                }
            )

    missing_start = [
        record
        for record in records
        if not record[
            "has_start"
        ]
    ]

    missing_end = [
        record
        for record in records
        if not record[
            "has_end"
        ]
    ]

    missing_start_with_end = [
        record
        for record in missing_start
        if record[
            "has_end"
        ]
    ]

    missing_start_without_end = [
        record
        for record in missing_start
        if not record[
            "has_end"
        ]
    ]

    missing_start_event_count = sum(
        record[
            "event_line_count"
        ]
        for record in missing_start
    )

    first_tokens = Counter(
        record[
            "first_token"
        ]
        for record in missing_start
    )

    summary = {
        "total_files":
            len(
                records
            ),

        "files_with_start":
            sum(
                record[
                    "has_start"
                ]
                for record in records
            ),

        "files_without_start":
            len(
                missing_start
            ),

        "files_with_end":
            sum(
                record[
                    "has_end"
                ]
                for record in records
            ),

        "files_without_end":
            len(
                missing_end
            ),

        "missing_start_with_end":
            len(
                missing_start_with_end
            ),

        "missing_start_without_end":
            len(
                missing_start_without_end
            ),

        "event_lines_in_missing_start_files":
            missing_start_event_count,

        "missing_start_first_tokens":
            dict(
                sorted(
                    first_tokens.items()
                )
            ),

        "missing_start_files":
            missing_start,
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
        "CLEMSON EVENT MARKER AUDIT"
    )
    print("=" * 60)

    print(
        f"Total files:                    "
        f"{summary['total_files']}"
    )

    print(
        f"Files with START:               "
        f"{summary['files_with_start']}"
    )

    print(
        f"Files without START:            "
        f"{summary['files_without_start']}"
    )

    print(
        f"Files with END:                 "
        f"{summary['files_with_end']}"
    )

    print(
        f"Files without END:              "
        f"{summary['files_without_end']}"
    )

    print(
        f"Missing START but has END:      "
        f"{summary['missing_start_with_end']}"
    )

    print(
        f"Missing START and missing END:  "
        f"{summary['missing_start_without_end']}"
    )

    print(
        f"Event lines in missing files:   "
        f"{summary['event_lines_in_missing_start_files']}"
    )

    print()

    print(
        "First tokens of files without START:"
    )

    for token, count in (
        summary[
            "missing_start_first_tokens"
        ].items()
    ):
        print(
            f"  {token:<25} {count}"
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