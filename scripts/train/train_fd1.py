import argparse
import copy
import csv
import time
from collections import defaultdict

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
    predict_sequence,
    save_json,
    set_seed,
)
from metrics import (
    confusion_matrix,
    match_segments,
    per_class_scores,
    prf,
    segments,
)
from models import GESTURE_MODELS, build_gesture_model
from postprocess import eating_speed, gesture_events


PROCESSED_DIR = ROOT / "data" / "processed" / "kuleuven_fd1"
WINDOWS_PATH = PROCESSED_DIR / "windows.csv"
CACHE_DIR = PROCESSED_DIR / "cache"
NORMALIZATION_PATH = ROOT / "config" / "fd1_normalization.json"

SAMPLE_RATE = 16
FULL_CHUNK = SAMPLE_RATE * 60 * 15
CLASS_NAMES = ["other", "eating", "drinking"]
EVENT_CLASSES = {"eating": 1, "drinking": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a KU Leuven FD-I per-sample gesture model."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["sqrt", "inverse", "none"], default="sqrt")
    parser.add_argument("--model", choices=sorted(GESTURE_MODELS), default="cnn")
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use only past readings (cnn only). Needed for the live demo.",
    )
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-event-seconds", type=float, default=0.5)
    parser.add_argument("--merge-gap-seconds", type=float, default=0.25)
    parser.add_argument("--bout-gap-seconds", type=float, default=60.0)
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


def class_weights(rows, mode):
    counts = np.array(
        [
            sum(int(row["other_samples"]) for row in rows),
            sum(int(row["eating_samples"]) for row in rows),
            sum(int(row["drinking_samples"]) for row in rows),
        ],
        dtype=np.float64,
    )

    if mode == "none":
        weights = np.ones(3)
    else:
        weights = counts.sum() / (len(counts) * np.maximum(counts, 1))

        if mode == "sqrt":
            weights = np.sqrt(weights)

    return torch.tensor(weights / weights[0], dtype=torch.float32)


def gesture_target(row, labels):
    return torch.from_numpy(labels.astype(np.int64))


def recordings(rows):
    seen = {}

    for row in rows:
        seen.setdefault(
            row["file"],
            (int(row["participant_id"]), row["wrist"]),
        )

    return sorted(seen.items())


def evaluate(model, rows, store, mean, std, device, args, predictions_path=None):
    min_length = round(args.min_event_seconds * SAMPLE_RATE)
    merge_gap = round(args.merge_gap_seconds * SAMPLE_RATE)
    context = model.context
    chunk = model.inference_chunk or FULL_CHUNK

    confusion = np.zeros((3, 3), dtype=np.int64)
    event_totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in EVENT_CLASSES}
    cross_matches = {"eating_predicted_as_drinking": 0, "drinking_predicted_as_eating": 0}
    per_file = []
    participant_bites = defaultdict(lambda: {"true": [], "predicted": []})
    prediction_rows = []

    for relative_file, (participant_id, wrist) in recordings(rows):
        X, y = store.get(relative_file)
        y = np.asarray(y)

        logits = predict_sequence(
            model,
            normalize(X, mean, std),
            device,
            chunk=chunk,
            context=context,
        )
        predicted = logits.argmax(dim=0).numpy().astype(np.uint8)

        confusion += confusion_matrix(y, predicted, 3)

        predicted_events = {}
        actual_events = {}

        for name, label in EVENT_CLASSES.items():
            predicted_events[name] = gesture_events(predicted, label, min_length, merge_gap)
            actual_events[name] = segments(y == label)

            match = match_segments(predicted_events[name], actual_events[name])

            for key in ("tp", "fp", "fn"):
                event_totals[name][key] += match[key]

            if predictions_path is not None:
                matched = {pair[0] for pair in match["pairs"]}

                for source, found in (("predicted", predicted_events[name]), ("true", actual_events[name])):
                    for start, end in found:
                        prediction_rows.append(
                            {
                                "participant_id": participant_id,
                                "wrist": wrist,
                                "event": name,
                                "source": source,
                                "start_seconds": start / SAMPLE_RATE,
                                "end_seconds": end / SAMPLE_RATE,
                                "matched": int(
                                    (start, end) in matched
                                    if source == "predicted"
                                    else any(pair[1] == (start, end) for pair in match["pairs"])
                                ),
                            }
                        )

        participant_bites[participant_id]["true"] += actual_events["eating"]
        participant_bites[participant_id]["predicted"] += predicted_events["eating"]

        cross_matches["eating_predicted_as_drinking"] += match_segments(
            predicted_events["drinking"], actual_events["eating"]
        )["tp"]
        cross_matches["drinking_predicted_as_eating"] += match_segments(
            predicted_events["eating"], actual_events["drinking"]
        )["tp"]

        per_file.append(
            {
                "file": relative_file,
                "participant_id": participant_id,
                "wrist": wrist,
                "true_bites": len(actual_events["eating"]),
                "predicted_bites": len(predicted_events["eating"]),
                "true_drinks": len(actual_events["drinking"]),
                "predicted_drinks": len(predicted_events["drinking"]),
            }
        )

    sample_scores = per_class_scores(confusion, CLASS_NAMES)

    if predictions_path is not None:
        write_rows(predictions_path, prediction_rows)

    return {
        "sample": {
            "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
            "class_names": CLASS_NAMES,
            "per_class": sample_scores,
            "macro_f1_eating_drinking": (
                sample_scores["eating"]["f1"] + sample_scores["drinking"]["f1"]
            ) / 2,
        },
        "events": {
            "min_event_seconds": args.min_event_seconds,
            "merge_gap_seconds": args.merge_gap_seconds,
            **{
                name: prf(**totals)
                for name, totals in event_totals.items()
            },
            "cross_class_matches": cross_matches,
        },
        "bite_count": bite_count_summary(per_file),
        "eating_speed": eating_speed_summary(participant_bites, args.bout_gap_seconds),
    }


def eating_speed_summary(participant_bites, bout_gap_seconds):
    participants = []

    for participant_id, bites in sorted(participant_bites.items()):
        true_speed = eating_speed(bites["true"], SAMPLE_RATE, bout_gap_seconds)
        predicted_speed = eating_speed(bites["predicted"], SAMPLE_RATE, bout_gap_seconds)
        participants.append(
            {
                "participant_id": participant_id,
                "true": true_speed,
                "predicted": predicted_speed,
            }
        )

    def error(key):
        pairs = [
            (item["predicted"][key], item["true"][key])
            for item in participants
            if item["predicted"][key] is not None and item["true"][key] is not None
        ]

        if not pairs:
            return {"mae": None, "participants_compared": 0}

        return {
            "mae": float(np.mean([abs(p - t) for p, t in pairs])),
            "participants_compared": len(pairs),
        }

    return {
        "bout_gap_seconds": bout_gap_seconds,
        "min_bites_per_bout": 3,
        "bites_per_minute_error": error("bites_per_minute"),
        "seconds_between_bites_error": error("median_seconds_between_bites"),
        "per_participant": participants,
    }


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        if not rows:
            return

        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bite_count_summary(per_file):
    by_participant = defaultdict(lambda: {"true_bites": 0, "predicted_bites": 0})

    for row in per_file:
        entry = by_participant[row["participant_id"]]
        entry["true_bites"] += row["true_bites"]
        entry["predicted_bites"] += row["predicted_bites"]

    participants = [
        {"participant_id": participant_id, **counts}
        for participant_id, counts in sorted(by_participant.items())
    ]

    def errors(items):
        absolute = [abs(item["predicted_bites"] - item["true_bites"]) for item in items]
        relative = [
            abs(item["predicted_bites"] - item["true_bites"]) / item["true_bites"]
            for item in items
            if item["true_bites"]
        ]
        return {
            "mae": float(np.mean(absolute)) if absolute else 0.0,
            "mean_relative_error": float(np.mean(relative)) if relative else 0.0,
        }

    return {
        "per_participant": participants,
        "per_participant_error": errors(participants),
        "per_file": per_file,
        "per_file_error": errors(per_file),
    }


def cpu_ms_per_minute(model, minutes=10, repeats=3):
    chunk = model.inference_chunk or FULL_CHUNK
    context = model.context
    cpu_model = copy.deepcopy(model).cpu().eval()
    data = np.random.default_rng(0).normal(
        size=(SAMPLE_RATE * 60 * minutes, 6)
    ).astype(np.float32)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    try:
        predict_sequence(cpu_model, data[:SAMPLE_RATE * 60], torch.device("cpu"), chunk, context)
        timings = []

        for _ in range(repeats):
            started = time.perf_counter()
            predict_sequence(cpu_model, data, torch.device("cpu"), chunk, context)
            timings.append(time.perf_counter() - started)
    finally:
        torch.set_num_threads(threads)

    return min(timings) * 1000 / minutes


def train_epoch(model, loader, optimizer, loss_fn, device):
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

    return total_loss / max(batches, 1)


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

    train_set = WindowDataset(rows["train"], store, mean, std, gesture_target)

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

    if args.causal and args.model != "cnn":
        raise SystemExit("--causal is only implemented for --model cnn")

    model_options = {"channels": args.channels, "dropout": args.dropout}

    if args.causal:
        model_options["causal"] = True

    model = build_gesture_model(args.model, **model_options).to(device)
    weights = class_weights(rows["train"], args.class_weight)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    run_dir = make_run_dir(
        f"fd1_{args.model}{'_causal' if args.causal else ''}", args.run_name
    )

    config = {
        **vars(args),
        "model_class": type(model).__name__,
        "parameters": count_parameters(model),
        "size_mb_float32": count_parameters(model) * 4 / 1e6,
        "context_samples": model.context,
        "lookahead_samples": model.lookahead,
        "lookahead_seconds": model.lookahead / SAMPLE_RATE,
        "inference_chunk_samples": model.inference_chunk or FULL_CHUNK,
        "class_weights": weights.tolist(),
        "device": str(device),
        "train_windows": len(rows["train"]),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
    }
    save_json(run_dir / "config.json", config)

    print()
    print(f"KU LEUVEN FD-I GESTURE MODEL: {args.model}")
    print("=" * 60)
    print(f"Run:            {run_dir.relative_to(ROOT)}")
    print(f"Device:         {device}")
    print(f"Parameters:     {config['parameters']:,}")
    print(f"Context:        {model.context} samples ({model.context / SAMPLE_RATE:.1f} s)")
    print(f"Look-ahead:     {model.lookahead / SAMPLE_RATE:.1f} s of future readings")
    print(f"Class weights:  {[round(w, 2) for w in weights.tolist()]}")
    print(f"Train windows:  {len(rows['train']):,}")
    print()

    history = []
    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        loss = train_epoch(model, loader, optimizer, loss_fn, device)
        validation = evaluate(model, rows["validation"], store, mean, std, device, args)
        score = validation["sample"]["macro_f1_eating_drinking"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "val_macro_f1": score,
                "val_eating_f1": validation["sample"]["per_class"]["eating"]["f1"],
                "val_drinking_f1": validation["sample"]["per_class"]["drinking"]["f1"],
                "val_bite_f1": validation["events"]["eating"]["f1"],
                "seconds": time.time() - started,
            }
        )
        save_json(run_dir / "history.json", history)

        improved = score > best_score

        if improved:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best.pt")

        print(
            f"epoch {epoch:3d}  loss {loss:.4f}  "
            f"val eat F1 {history[-1]['val_eating_f1']:.3f}  "
            f"drink F1 {history[-1]['val_drinking_f1']:.3f}  "
            f"bite F1 {history[-1]['val_bite_f1']:.3f}  "
            f"{history[-1]['seconds']:.0f}s"
            f"{'  *' if improved else ''}"
        )

        if epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping.")
            break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))

    results = {
        "best_epoch": best_epoch,
        "model": args.model + ("_causal" if args.causal else ""),
        "lookahead_seconds": config["lookahead_seconds"],
        "parameters": config["parameters"],
        "size_mb_float32": config["size_mb_float32"],
        "cpu_ms_per_minute_single_thread": cpu_ms_per_minute(model),
    }
    splits = ["validation"] + (["test"] if args.eval_test else [])

    for split_name in splits:
        results[split_name] = evaluate(
            model,
            rows[split_name],
            store,
            mean,
            std,
            device,
            args,
            predictions_path=run_dir / f"predictions_{split_name}.csv",
        )

    save_json(run_dir / "metrics.json", results)

    print()
    print(f"Best epoch: {best_epoch}")
    print(f"CPU time:   {results['cpu_ms_per_minute_single_thread']:.1f} ms "
          f"per minute of data (1 thread)")

    for split_name in splits:
        metrics = results[split_name]
        print()
        print(split_name.upper())
        print(f"  sample F1   eating {metrics['sample']['per_class']['eating']['f1']:.3f}  "
              f"drinking {metrics['sample']['per_class']['drinking']['f1']:.3f}")
        print(f"  event F1    bites {metrics['events']['eating']['f1']:.3f}  "
              f"drinks {metrics['events']['drinking']['f1']:.3f}")
        print(f"  bite count  MAE per participant "
              f"{metrics['bite_count']['per_participant_error']['mae']:.1f}  "
              f"relative {metrics['bite_count']['per_participant_error']['mean_relative_error']:.1%}")

        speed = metrics["eating_speed"]
        rate_error = speed["bites_per_minute_error"]
        gap_error = speed["seconds_between_bites_error"]

        if rate_error["mae"] is not None:
            print(f"  eating speed  bites/min MAE {rate_error['mae']:.2f}  "
                  f"time between bites MAE {gap_error['mae']:.1f} s  "
                  f"({rate_error['participants_compared']} participants)")

    print()
    print(f"Saved: {(run_dir / 'metrics.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
