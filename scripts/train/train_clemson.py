import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from common import (
    ROOT,
    RecordingStore,
    WindowDataset,
    count_parameters,
    load_normalization,
    load_windows,
    make_run_dir,
    normalize,
    pick_device,
    save_json,
    set_seed,
)
from metrics import (
    match_segments,
    overlap_length,
    prf,
    segments,
)
from models import EpisodeCNN
from postprocess import reconstruct_meals, segment_probabilities


PROCESSED_DIR = ROOT / "data" / "processed" / "clemson_allday"
WINDOWS_PATH = PROCESSED_DIR / "windows.csv"
CACHE_DIR = PROCESSED_DIR / "cache"
NORMALIZATION_PATH = ROOT / "config" / "clemson_normalization.json"

SAMPLE_RATE = 15

THRESHOLD_GRID = [round(value, 2) for value in np.arange(0.1, 0.96, 0.05)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Clemson All-day window CNN and rebuild eating episodes."
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", default="sqrt", help="'sqrt', 'none' or a number")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--merge-gap-minutes", type=float, default=3.0)
    parser.add_argument("--min-episode-minutes", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Also score the test split. Use only for the final reported model.",
    )
    return parser.parse_args()


def fraction_target(row, labels):
    window = int(row["end_sample"]) - int(row["start_sample"])
    return torch.tensor(int(row["eating_samples"]) / window, dtype=torch.float32)


def positive_weight(rows, mode):
    if mode == "none":
        return 1.0

    if mode != "sqrt":
        return float(mode)

    eating = sum(int(row["eating_samples"]) for row in rows)
    total = sum(int(row["eating_samples"]) + int(row["other_samples"]) for row in rows)
    fraction = max(eating / total, 1e-6)
    return float(np.sqrt((1 - fraction) / fraction))


def group_by_recording(rows):
    grouped = {}

    for row in rows:
        grouped.setdefault(row["file"], []).append(row)

    return dict(sorted(grouped.items()))


@torch.no_grad()
def window_probabilities(model, grouped, store, mean, std, device, batch_size):
    model.eval()
    results = {}

    for relative_file, recording_rows in grouped.items():
        X, y = store.get(relative_file)
        X = normalize(X, mean, std)
        starts = [int(row["start_sample"]) for row in recording_rows]
        length = int(recording_rows[0]["end_sample"]) - starts[0]
        probabilities = []

        for offset in range(0, len(starts), batch_size):
            batch = np.stack(
                [X[start:start + length].T for start in starts[offset:offset + batch_size]]
            )
            logits = model(torch.from_numpy(batch).to(device))
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy())

        results[relative_file] = {
            "starts": np.array(starts),
            "window_samples": length,
            "probabilities": np.concatenate(probabilities),
            "fractions": np.array(
                [int(row["eating_samples"]) / length for row in recording_rows]
            ),
            "labels": np.asarray(y),
        }

    return results


def window_scores(predictions, threshold=0.5):
    probabilities = np.concatenate([item["probabilities"] for item in predictions.values()])
    fractions = np.concatenate([item["fractions"] for item in predictions.values()])
    predicted = probabilities >= threshold
    actual = fractions >= 0.5

    eps = 1e-7
    loss = -np.mean(
        fractions * np.log(probabilities + eps)
        + (1 - fractions) * np.log(1 - probabilities + eps)
    )

    return {
        "loss": float(loss),
        **prf(
            int((predicted & actual).sum()),
            int((predicted & ~actual).sum()),
            int((~predicted & actual).sum()),
        ),
    }


def predicted_meals(item, stride, start_threshold, end_threshold, args):
    total = len(item["labels"])
    timeline = segment_probabilities(
        item["starts"], item["probabilities"], item["window_samples"], total, stride
    )
    return reconstruct_meals(
        timeline,
        stride,
        total,
        start_threshold,
        end_threshold,
        round(args.merge_gap_minutes * 60 * SAMPLE_RATE),
        round(args.min_episode_minutes * 60 * SAMPLE_RATE),
    )


def episode_scores(
    predictions, stride, start_threshold, end_threshold, args, meals_path=None
):
    tp = fp = fn = 0
    total_samples = 0
    time_tp = time_fp = time_fn = 0
    start_errors = []
    end_errors = []
    per_recording = []
    meal_rows = []

    for relative_file, item in predictions.items():
        labels = item["labels"]
        total = len(labels)
        total_samples += total

        predicted = predicted_meals(item, stride, start_threshold, end_threshold, args)
        actual = segments(labels == 1)
        match = match_segments(predicted, actual)

        tp += match["tp"]
        fp += match["fp"]
        fn += match["fn"]

        for (pred_start, pred_end), (true_start, true_end) in match["pairs"]:
            start_errors.append(abs(pred_start - true_start) / SAMPLE_RATE / 60)
            end_errors.append(abs(pred_end - true_end) / SAMPLE_RATE / 60)

        overlap = overlap_length(predicted, actual)
        time_tp += overlap
        time_fp += sum(end - start for start, end in predicted) - overlap
        time_fn += sum(end - start for start, end in actual) - overlap

        recording_id = Path(relative_file).stem

        per_recording.append(
            {
                "recording_id": recording_id,
                "file": relative_file,
                "hours": total / SAMPLE_RATE / 3600,
                "true_meals": len(actual),
                "predicted_meals": len(predicted),
                "detected": match["tp"],
                "false_alarms": match["fp"],
                "true_eating_minutes": sum(e - s for s, e in actual) / SAMPLE_RATE / 60,
                "predicted_eating_minutes": sum(e - s for s, e in predicted) / SAMPLE_RATE / 60,
            }
        )

        if meals_path is not None:
            partner = {}
            for pred, true in match["pairs"]:
                partner[("predicted", pred)] = true
                partner[("true", true)] = pred

            for source, found in (("predicted", predicted), ("true", actual)):
                for start, end in found:
                    other = partner.get((source, (start, end)))
                    meal_rows.append(
                        {
                            "recording_id": recording_id,
                            "source": source,
                            "start_seconds": start / SAMPLE_RATE,
                            "end_seconds": end / SAMPLE_RATE,
                            "matched": int(other is not None),
                            "matched_start_seconds": other[0] / SAMPLE_RATE if other else "",
                            "matched_end_seconds": other[1] / SAMPLE_RATE if other else "",
                        }
                    )

    if meals_path is not None:
        write_rows(meals_path, meal_rows)

    hours = total_samples / SAMPLE_RATE / 3600
    episodes = prf(tp, fp, fn)

    return {
        "start_threshold": start_threshold,
        "end_threshold": end_threshold,
        "merge_gap_minutes": args.merge_gap_minutes,
        "min_episode_minutes": args.min_episode_minutes,
        "hours": hours,
        "episodes": {
            **episodes,
            "true_positive_rate": episodes["recall"],
            "false_alarms_per_hour": fp / hours if hours else 0.0,
            "false_alarms_per_day": fp / hours * 24 if hours else 0.0,
            "mean_start_error_minutes": float(np.mean(start_errors)) if start_errors else None,
            "mean_end_error_minutes": float(np.mean(end_errors)) if end_errors else None,
        },
        "time": prf(time_tp, time_fp, time_fn),
        "daily_totals": daily_totals(per_recording),
        "per_recording": per_recording,
    }


def daily_totals(per_recording):
    if not per_recording:
        return {}

    def mae(true_key, predicted_key):
        return float(np.mean([abs(row[predicted_key] - row[true_key]) for row in per_recording]))

    return {
        "days": len(per_recording),
        "mean_true_meals_per_day": float(np.mean([row["true_meals"] for row in per_recording])),
        "mean_predicted_meals_per_day": float(np.mean([row["predicted_meals"] for row in per_recording])),
        "meals_per_day_mae": mae("true_meals", "predicted_meals"),
        "mean_true_eating_minutes_per_day": float(np.mean([row["true_eating_minutes"] for row in per_recording])),
        "mean_predicted_eating_minutes_per_day": float(np.mean([row["predicted_eating_minutes"] for row in per_recording])),
        "eating_minutes_per_day_mae": mae("true_eating_minutes", "predicted_eating_minutes"),
    }


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        if not rows:
            return

        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tune_thresholds(predictions, stride, args):
    best = None

    for start_threshold in THRESHOLD_GRID:
        for end_threshold in THRESHOLD_GRID:
            if end_threshold > start_threshold:
                continue

            scores = episode_scores(predictions, stride, start_threshold, end_threshold, args)
            key = (scores["episodes"]["f1"], scores["time"]["f1"])

            if best is None or key > best[0]:
                best = (key, scores)

    return best[1]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    mean, std = load_normalization(NORMALIZATION_PATH)
    store = RecordingStore(CACHE_DIR)

    rows = {
        split_name: load_windows(WINDOWS_PATH, split_name)
        for split_name in ("train", "validation", "test")
    }

    first = rows["train"][0]
    window_samples = int(first["end_sample"]) - int(first["start_sample"])
    stride = None

    for previous, current in zip(rows["train"], rows["train"][1:]):
        if previous["file"] == current["file"]:
            stride = int(current["start_sample"]) - int(previous["start_sample"])
            break

    if stride is None or window_samples % stride:
        raise ValueError("Window length must be a whole number of strides.")

    train_set = WindowDataset(rows["train"], store, mean, std, fraction_target)

    sampler = None
    if args.max_train_windows:
        sampler = RandomSampler(
            train_set,
            replacement=False,
            num_samples=min(args.max_train_windows, len(train_set)),
        )

    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = EpisodeCNN(dropout=args.dropout).to(device)
    pos_weight = positive_weight(rows["train"], args.pos_weight)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    validation_groups = group_by_recording(rows["validation"])
    run_dir = make_run_dir("clemson_cnn", args.run_name)

    config = {
        **vars(args),
        "model": "EpisodeCNN",
        "parameters": count_parameters(model),
        "window_samples": window_samples,
        "stride_samples": stride,
        "pos_weight_value": pos_weight,
        "device": str(device),
        "train_windows": len(rows["train"]),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
    }
    save_json(run_dir / "config.json", config)

    print()
    print("CLEMSON ALL-DAY EPISODE CNN")
    print("=" * 60)
    print(f"Run:           {run_dir.relative_to(ROOT)}")
    print(f"Device:        {device}")
    print(f"Parameters:    {config['parameters']:,}")
    print(f"Window/stride: {window_samples / SAMPLE_RATE / 60:.0f} / "
          f"{stride / SAMPLE_RATE / 60:.0f} min")
    print(f"pos_weight:    {pos_weight:.2f}")
    print(f"Train windows: {len(rows['train']):,}")
    print()

    history = []
    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        total_loss = 0.0
        batches = 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        predictions = window_probabilities(
            model, validation_groups, store, mean, std, device, args.batch_size
        )
        validation = window_scores(predictions)

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(batches, 1),
                "val_loss": validation["loss"],
                "val_window_f1": validation["f1"],
                "seconds": time.time() - started,
            }
        )
        save_json(run_dir / "history.json", history)

        improved = validation["f1"] > best_score

        if improved:
            best_score = validation["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best.pt")

        print(
            f"epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  "
            f"val loss {validation['loss']:.4f}  "
            f"val window F1 {validation['f1']:.3f}  "
            f"{history[-1]['seconds']:.0f}s"
            f"{'  *' if improved else ''}"
        )

        if epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping.")
            break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))

    validation_predictions = window_probabilities(
        model, validation_groups, store, mean, std, device, args.batch_size
    )
    tuned = tune_thresholds(validation_predictions, stride, args)
    tuned = episode_scores(
        validation_predictions,
        stride,
        tuned["start_threshold"],
        tuned["end_threshold"],
        args,
        meals_path=run_dir / "meals_validation.csv",
    )

    results = {
        "best_epoch": best_epoch,
        "thresholds_tuned_on": "validation",
        "validation": {
            "windows": window_scores(validation_predictions),
            **tuned,
        },
    }

    if args.eval_test:
        test_predictions = window_probabilities(
            model, group_by_recording(rows["test"]), store, mean, std, device, args.batch_size
        )
        results["test"] = {
            "windows": window_scores(test_predictions),
            **episode_scores(
                test_predictions,
                stride,
                tuned["start_threshold"],
                tuned["end_threshold"],
                args,
                meals_path=run_dir / "meals_test.csv",
            ),
        }

    save_json(run_dir / "metrics.json", results)

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Thresholds (from validation): start {tuned['start_threshold']}, "
          f"end {tuned['end_threshold']}")

    for split_name in ("validation", "test"):
        if split_name not in results:
            continue

        metrics = results[split_name]
        episodes = metrics["episodes"]
        print()
        print(split_name.upper())
        print(f"  window F1          {metrics['windows']['f1']:.3f}")
        print(f"  episodes found     {episodes['tp']}/{episodes['tp'] + episodes['fn']} "
              f"({episodes['true_positive_rate']:.1%})")
        print(f"  false alarms       {episodes['fp']} "
              f"({episodes['false_alarms_per_hour']:.3f}/h, "
              f"{episodes['false_alarms_per_day']:.2f}/day)")
        print(f"  episode F1         {episodes['f1']:.3f}")
        print(f"  time F1            {metrics['time']['f1']:.3f}")
        totals = metrics["daily_totals"]
        print(f"  meals per day      true {totals['mean_true_meals_per_day']:.2f}  "
              f"predicted {totals['mean_predicted_meals_per_day']:.2f}  "
              f"MAE {totals['meals_per_day_mae']:.2f}")
        print(f"  eating min per day true {totals['mean_true_eating_minutes_per_day']:.1f}  "
              f"predicted {totals['mean_predicted_eating_minutes_per_day']:.1f}  "
              f"MAE {totals['eating_minutes_per_day_mae']:.1f}")

    print()
    print(f"Saved: {(run_dir / 'metrics.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
