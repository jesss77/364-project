import argparse
import csv

from common import ROOT, RUNS_DIR, load_json


OUTPUT_PATH = ROOT / "results" / "fd1_model_comparison.csv"


def split_columns(metrics, prefix):
    if metrics is None:
        return {}

    per_class = metrics["sample"]["per_class"]
    events = metrics["events"]
    bites = metrics["bite_count"]["per_participant_error"]

    return {
        f"{prefix}_eating_f1": per_class["eating"]["f1"],
        f"{prefix}_drinking_f1": per_class["drinking"]["f1"],
        f"{prefix}_bite_event_f1": events["eating"]["f1"],
        f"{prefix}_drink_event_f1": events["drinking"]["f1"],
        f"{prefix}_bite_count_mae": bites["mae"],
        f"{prefix}_bite_count_relative_error": bites["mean_relative_error"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect KU Leuven runs from results/runs into one table."
    )
    parser.add_argument("runs", nargs="*", help="Run names; default is every fd1 run")
    args = parser.parse_args()

    paths = (
        [RUNS_DIR / name / "metrics.json" for name in args.runs]
        if args.runs
        else sorted(RUNS_DIR.glob("*/metrics.json"))
    )

    rows = []

    for path in paths:
        metrics = load_json(path)
        config_path = path.parent / "config.json"

        if "bite_count" not in metrics.get("validation", {}):
            continue

        config = load_json(config_path) if config_path.exists() else {}

        rows.append(
            {
                "run": path.parent.name,
                "model": metrics.get("model", config.get("model", "cnn")),
                "parameters": metrics.get("parameters", config.get("parameters")),
                "size_mb_float32": metrics.get("size_mb_float32"),
                "cpu_ms_per_minute": metrics.get("cpu_ms_per_minute_single_thread"),
                "best_epoch": metrics["best_epoch"],
                "seed": config.get("seed"),
                **split_columns(metrics["validation"], "val"),
                **split_columns(metrics.get("test"), "test"),
            }
        )

    if not rows:
        raise SystemExit(f"No KU Leuven runs found under {RUNS_DIR.relative_to(ROOT)}")

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    header = (
        f"{'run':<28} {'model':<12} {'params':>8} {'ms/min':>7} "
        f"{'eat F1':>7} {'drink F1':>8} {'bite F1':>8} {'bite MAE':>9}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        cpu = row["cpu_ms_per_minute"]
        print(
            f"{row['run']:<28} {row['model']:<12} {row['parameters'] or 0:>8,} "
            f"{cpu if cpu is not None else float('nan'):>7.1f} "
            f"{row['val_eating_f1']:>7.3f} {row['val_drinking_f1']:>8.3f} "
            f"{row['val_bite_event_f1']:>8.3f} {row['val_bite_count_mae']:>9.1f}"
        )

    print()
    print("Scores are on the validation split.")
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
