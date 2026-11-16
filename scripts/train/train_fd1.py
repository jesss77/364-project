import argparse
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
    clean_segments,
    confusion_matrix,
    match_segments,
    per_class_scores,
    prf,
    segments,
)
from models import GestureCNN


PROCESSED_DIR = ROOT / "data" / "processed" / "kuleuven_fd1"
WINDOWS_PATH = PROCESSED_DIR / "windows.csv"
CACHE_DIR = PROCESSED_DIR / "cache"
NORMALIZATION_PATH = ROOT / "config" / "fd1_normalization.json"

SAMPLE_RATE = 16
CLASS_NAMES = ["other", "eating", "drinking"]
EVENT_CLASSES = {"eating": 1, "drinking": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the KU Leuven FD-I per-sample gesture CNN."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["sqrt", "inverse", "none"], default="sqrt")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-event-seconds", type=float, default=0.5)
    parser.add_argument("--merge-gap-seconds", type=float, default=0.25)
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


def evaluate(model, rows, store, mean, std, device, args):
    min_length = round(args.min_event_seconds * SAMPLE_RATE)
    merge_gap = round(args.merge_gap_seconds * SAMPLE_RATE)
    context = model.receptive_field

    confusion = np.zeros((3, 3), dtype=np.int64)
    event_totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in EVENT_CLASSES}
    cross_matches = {"eating_predicted_as_drinking": 0, "drinking_predicted_as_eating": 0}
    per_file = []

    for relative_file, (participant_id, wrist) in recordings(rows):
        X, y = store.get(relative_file)
        y = np.asarray(y)

        logits = predict_sequence(
            model,
            normalize(X, mean, std),
            device,
            chunk=SAMPLE_RATE * 60 * 15,
            context=context,
        )
        predicted = logits.argmax(dim=0).numpy().astype(np.uint8)

        confusion += confusion_matrix(y, predicted, 3)

        predicted_events = {}
        actual_events = {}

        for name, label in EVENT_CLASSES.items():
            predicted_events[name] = clean_segments(
                segments(predicted == label), merge_gap, min_length
            )
            actual_events[name] = segments(y == label)

            match = match_segments(predicted_events[name], actual_events[name])

            for key in ("tp", "fp", "fn"):
                event_totals[name][key] += match[key]

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
    }


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

    model = GestureCNN(channels=args.channels, dropout=args.dropout).to(device)
    weights = class_weights(rows["train"], args.class_weight)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    run_dir = make_run_dir("fd1_cnn", args.run_name)

    config = {
        **vars(args),
        "model": "GestureCNN",
        "parameters": count_parameters(model),
        "receptive_field_samples": model.receptive_field,
        "class_weights": weights.tolist(),
        "device": str(device),
        "train_windows": len(rows["train"]),
    }
    save_json(run_dir / "config.json", config)

    print()
    print("KU LEUVEN FD-I GESTURE CNN")
    print("=" * 60)
    print(f"Run:            {run_dir.relative_to(ROOT)}")
    print(f"Device:         {device}")
    print(f"Parameters:     {config['parameters']:,}")
    print(f"Receptive field {model.receptive_field} samples "
          f"({model.receptive_field / SAMPLE_RATE:.1f} s)")
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

    results = {"best_epoch": best_epoch}
    splits = ["validation"] + (["test"] if args.eval_test else [])

    for split_name in splits:
        results[split_name] = evaluate(model, rows[split_name], store, mean, std, device, args)

    save_json(run_dir / "metrics.json", results)

    print()
    print(f"Best epoch: {best_epoch}")

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

    print()
    print(f"Saved: {(run_dir / 'metrics.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
