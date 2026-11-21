import argparse
import csv
import re
from collections import defaultdict

import numpy as np
from openpyxl import load_workbook

from common import ROOT, RUNS_DIR, load_json, save_json


EVENTS_CSV = ROOT / "data" / "processed" / "clemson_allday" / "events.csv"
DEMOGRAPHICS_FILE = ROOT / "data" / "raw" / "clemson_allday" / "ShimmerDemographics.xlsx"

MIN_GROUP_EVENTS = 10

DEMOGRAPHIC_COLUMNS = {
    "age": "age",
    "gender": "gender",
    "bmi": "bmi",
    "handedness (which hand do you eat with?)": "eating_hand",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Break down Clemson meal detection by meal details and demographics."
    )
    parser.add_argument("run", help="Run name under results/runs")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument(
        "--min-group",
        type=int,
        default=MIN_GROUP_EVENTS,
        help="Groups with fewer events are merged into 'other (small groups)'",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def meal_type(name):
    name = re.sub(r"[^a-z]", "", name.lower())

    if "snack" in name:
        return "snack"
    if "breakfast" in name or "brunch" in name:
        return "breakfast"
    if "lunch" in name:
        return "lunch"
    if "dinner" in name:
        return "dinner"

    return "other"


def duration_bucket(minutes):
    if minutes < 5:
        return "under 5 min"
    if minutes < 15:
        return "5-15 min"
    if minutes < 30:
        return "15-30 min"
    return "30+ min"


def bmi_bucket(value):
    if value is None:
        return "unknown"
    if value < 18.5:
        return "under 18.5"
    if value < 25:
        return "18.5-25"
    if value < 30:
        return "25-30"
    return "30+"


def age_bucket(value):
    if value is None:
        return "unknown"
    if value < 25:
        return "under 25"
    if value < 40:
        return "25-40"
    if value < 60:
        return "40-60"
    return "60+"


def to_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_demographics():
    if not DEMOGRAPHICS_FILE.exists():
        return {}

    workbook = load_workbook(DEMOGRAPHICS_FILE, read_only=True, data_only=True)
    people = {}

    for sheet in workbook.worksheets:
        rows = sheet.iter_rows(values_only=True)
        header = [str(value).strip().lower() if value is not None else "" for value in next(rows, [])]

        if "participant id" not in header:
            continue

        id_column = header.index("participant id")

        for row in rows:
            if id_column >= len(row) or not row[id_column]:
                continue

            record = {}

            for column, key in DEMOGRAPHIC_COLUMNS.items():
                if column in header and header.index(column) < len(row):
                    record[key] = row[header.index(column)]

            people[str(row[id_column]).strip().upper()] = record

    workbook.close()
    return people


def attach_events(meals, events):
    true_meals = defaultdict(list)

    for meal in meals:
        if meal["source"] == "true":
            true_meals[meal["recording_id"]].append(meal)

    attached = []

    for event in events:
        start = float(event["start_offset_seconds"])
        end = float(event["end_offset_seconds"])
        best = None
        best_overlap = 0.0

        for meal in true_meals.get(event["recording_id"], []):
            overlap = min(end, float(meal["end_seconds"])) - max(start, float(meal["start_seconds"]))

            if overlap > best_overlap:
                best = meal
                best_overlap = overlap

        if best is None:
            continue

        matched = best["matched"] == "1"
        attached.append(
            {
                **event,
                "detected": matched,
                "start_error_minutes": (
                    abs(float(best["matched_start_seconds"]) - float(best["start_seconds"])) / 60
                    if matched
                    else None
                ),
                "end_error_minutes": (
                    abs(float(best["matched_end_seconds"]) - float(best["end_seconds"])) / 60
                    if matched
                    else None
                ),
            }
        )

    return attached


def summarize(items, key_fn, min_group=MIN_GROUP_EVENTS):
    groups = defaultdict(list)

    for item in items:
        groups[key_fn(item)].append(item)

    small = [key for key, members in groups.items() if len(members) < min_group]

    if len(small) > 1:
        merged = []
        for key in small:
            merged += groups.pop(key)
        groups["other (small groups)"] = merged

    table = []

    for key, members in sorted(groups.items(), key=lambda pair: -len(pair[1])):
        starts = [m["start_error_minutes"] for m in members if m["start_error_minutes"] is not None]
        ends = [m["end_error_minutes"] for m in members if m["end_error_minutes"] is not None]
        detected = sum(m["detected"] for m in members)
        table.append(
            {
                "group": str(key),
                "events": len(members),
                "detected": detected,
                "detection_rate": detected / len(members),
                "mean_start_error_minutes": float(np.mean(starts)) if starts else None,
                "mean_end_error_minutes": float(np.mean(ends)) if ends else None,
            }
        )

    return table


def recording_summary(per_recording, people, key_fn):
    groups = defaultdict(lambda: {"recordings": 0, "hours": 0.0, "true": 0, "detected": 0, "false_alarms": 0})

    for row in per_recording:
        group = groups[key_fn(people.get(row["recording_id"], {}))]
        group["recordings"] += 1
        group["hours"] += row["hours"]
        group["true"] += row["true_meals"]
        group["detected"] += row["detected"]
        group["false_alarms"] += row["false_alarms"]

    return [
        {
            "group": key,
            "recordings": group["recordings"],
            "meals": group["true"],
            "detection_rate": group["detected"] / group["true"] if group["true"] else None,
            "false_alarms_per_day": group["false_alarms"] / group["hours"] * 24 if group["hours"] else None,
        }
        for key, group in sorted(groups.items(), key=lambda pair: -pair[1]["recordings"])
    ]


def print_table(title, rows, columns):
    print()
    print(title)
    print("-" * 72)

    for row in rows:
        parts = [f"{str(row['group'])[:28]:<28}"]

        for column, label in columns:
            value = row[column]

            if value is None:
                parts.append(f"{label} -")
            elif isinstance(value, float):
                parts.append(f"{label} {value:.2f}")
            else:
                parts.append(f"{label} {value}")

        print("  ".join(parts))


def main():
    args = parse_args()
    run_dir = RUNS_DIR / args.run
    meals_path = run_dir / f"meals_{args.split}.csv"

    if not meals_path.exists():
        raise SystemExit(
            f"{meals_path.relative_to(ROOT)} not found. "
            "Train with train_clemson.py first (use --eval-test for the test split)."
        )

    meals = read_csv(meals_path)
    metrics = load_json(run_dir / "metrics.json")[args.split]
    recording_ids = {row["recording_id"] for row in metrics["per_recording"]}
    events = [row for row in read_csv(EVENTS_CSV) if row["recording_id"] in recording_ids]
    people = load_demographics()

    attached = attach_events(meals, events)

    def demographic(name, bucket=None):
        def key(item):
            value = people.get(item["recording_id"], {}).get(name)
            if bucket:
                return bucket(to_number(value))
            return str(value).strip() if value not in (None, "") else "unknown"
        return key

    event_tables = {
        "meal_type": summarize(attached, lambda e: meal_type(e.get("event_type", "")), args.min_group),
        "duration": summarize(attached, lambda e: duration_bucket(float(e["duration_seconds"]) / 60), args.min_group),
        "location": summarize(attached, lambda e: e.get("location") or "unknown", args.min_group),
        "social_context": summarize(attached, lambda e: e.get("social_context") or "unknown", args.min_group),
        "activity": summarize(attached, lambda e: e.get("activity") or "unknown", args.min_group),
        "time_precision": summarize(attached, lambda e: e.get("seconds_status") or "unknown", args.min_group),
        "bmi": summarize(attached, demographic("bmi", bmi_bucket), args.min_group),
        "age": summarize(attached, demographic("age", age_bucket), args.min_group),
        "gender": summarize(attached, demographic("gender"), args.min_group),
        "eating_hand": summarize(attached, demographic("eating_hand"), args.min_group),
    }

    def person_key(name, bucket=None):
        def key(person):
            value = person.get(name)
            if bucket:
                return bucket(to_number(value))
            return str(value).strip() if value not in (None, "") else "unknown"
        return key

    false_alarm_tables = {
        "bmi": recording_summary(metrics["per_recording"], people, person_key("bmi", bmi_bucket)),
        "age": recording_summary(metrics["per_recording"], people, person_key("age", age_bucket)),
        "gender": recording_summary(metrics["per_recording"], people, person_key("gender")),
        "eating_hand": recording_summary(metrics["per_recording"], people, person_key("eating_hand")),
    }

    output = {
        "run": args.run,
        "split": args.split,
        "events": len(attached),
        "events_without_true_meal": len(events) - len(attached),
        "demographics_found": sum(1 for rid in recording_ids if rid in people),
        "recordings": len(recording_ids),
        "by_event": event_tables,
        "by_person": false_alarm_tables,
    }

    output_path = run_dir / f"meal_breakdown_{args.split}.json"
    save_json(output_path, output)

    print(f"CLEMSON MEAL BREAKDOWN  run {args.run}  split {args.split}")
    print(f"Events matched to labels: {len(attached)} of {len(events)}")
    print(f"Recordings with demographics: {output['demographics_found']} of {len(recording_ids)}")

    event_columns = [
        ("events", "n"),
        ("detection_rate", "found"),
        ("mean_start_error_minutes", "start err"),
        ("mean_end_error_minutes", "end err"),
    ]

    for name, rows in event_tables.items():
        print_table(f"By {name.replace('_', ' ')}", rows, event_columns)

    person_columns = [
        ("recordings", "days"),
        ("detection_rate", "found"),
        ("false_alarms_per_day", "false/day"),
    ]

    for name, rows in false_alarm_tables.items():
        print_table(f"False alarms by {name.replace('_', ' ')}", rows, person_columns)

    print()
    print(f"Saved: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
