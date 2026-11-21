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
    load_json,
    load_normalization,
    make_run_dir,
    normalize,
    pick_device,
    predict_sequence,
    save_json,
    set_seed,
)
from metrics import confusion_matrix, match_segments, per_class_scores, prf, segments
from models import GESTURE_MODELS, build_gesture_model
from postprocess import gesture_events
from train_fd1 import (
    CACHE_DIR,
    CLASS_NAMES,
    FULL_CHUNK,
    NORMALIZATION_PATH,
    SAMPLE_RATE,
    WINDOWS_PATH,
    class_weights,
    gesture_target,
    train_epoch,
)


FOLDS_PATH = ROOT / "config" / "fd1_folds.json"

WINDOW_SAMPLES = 960
ADAPT_STRIDE = 128
LEAD_IN_SECONDS = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Q5: train on the other folds, then fine-tune on a few minutes of a held-out "
            "person's labeled data and score the rest of their recording."
        )
    )
    parser.add_argument("--model", choices=sorted(GESTURE_MODELS), default="cnn")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*", help="Fold indices to run; default all")
    parser.add_argument("--folds-file", default=str(FOLDS_PATH))
    parser.add_argument(
        "--random-folds",
        type=int,
        default=0,
        help="Make this many seeded folds instead of reading the folds file (testing only)",
    )
    parser.add_argument("--budgets", type=float, nargs="+", default=[1, 2, 5])
    parser.add_argument("--modes", nargs="+", choices=["head", "all"], default=["head", "all"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adapt-epochs", type=int, default=30)
    parser.add_argument("--adapt-lr-head", type=float, default=1e-3)
    parser.add_argument("--adapt-lr-all", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["sqrt", "inverse", "none"], default="sqrt")
    parser.add_argument("--min-event-seconds", type=float, default=0.5)
    parser.add_argument("--merge-gap-seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def read_all_windows():
    with WINDOWS_PATH.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_folds(args, participants):
    if args.random_folds:
        rng = np.random.default_rng(args.seed)
        shuffled = rng.permutation(sorted(participants)).tolist()
        return [sorted(shuffled[index::args.random_folds]) for index in range(args.random_folds)], "random (testing only)"

    folds = load_json(args.folds_file)
    return [sorted(int(p) for p in fold) for fold in folds["folds"]], folds.get("source", args.folds_file)


def build_model(args):
    options = {"causal": True} if args.causal else {}
    return build_gesture_model(args.model, **options)


def train_base(rows, store, mean, std, device, args):
    dataset = WindowDataset(rows, store, mean, std, gesture_target)
    sampler = None

    if args.max_train_windows:
        sampler = RandomSampler(dataset, num_samples=min(args.max_train_windows, len(dataset)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
    )
    model = build_model(args).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(rows, args.class_weight).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for _ in range(args.epochs):
        train_epoch(model, loader, optimizer, loss_fn, device)

    return model, loss_fn


def adaptation_start(labels_by_file):
    firsts = [
        int(np.flatnonzero(labels == 1)[0])
        for labels in labels_by_file.values()
        if (labels == 1).any()
    ]

    if not firsts:
        return None

    return max(min(firsts) - LEAD_IN_SECONDS * SAMPLE_RATE, 0)


def adaptation_windows(files, start, budget_samples):
    rows = []

    for relative_file, labels in files.items():
        end = min(start + budget_samples, len(labels))

        for window_start in range(start, end - WINDOW_SAMPLES + 1, ADAPT_STRIDE):
            rows.append(
                {
                    "file": relative_file,
                    "start_sample": window_start,
                    "end_sample": window_start + WINDOW_SAMPLES,
                }
            )

    return rows


def fine_tune(base, rows, store, mean, std, loss_fn, device, mode, args):
    model = copy.deepcopy(base)

    for name, parameter in model.named_parameters():
        parameter.requires_grad = mode == "all" or name.startswith("head.")

    lr = args.adapt_lr_all if mode == "all" else args.adapt_lr_head
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=lr
    )
    loader = DataLoader(
        WindowDataset(rows, store, mean, std, gesture_target),
        batch_size=16,
        shuffle=True,
    )

    for _ in range(args.adapt_epochs):
        model.train()

        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()

    return model.eval()


def score_region(model, files, store, mean, std, cut, device, args):
    min_length = round(args.min_event_seconds * SAMPLE_RATE)
    merge_gap = round(args.merge_gap_seconds * SAMPLE_RATE)
    confusion = np.zeros((3, 3), dtype=np.int64)
    bites = {"tp": 0, "fp": 0, "fn": 0}
    true_bites = predicted_bites = 0

    for relative_file in files:
        X, y = store.get(relative_file)
        y = np.asarray(y)[cut:]
        logits = predict_sequence(
            model,
            normalize(X, mean, std),
            device,
            chunk=model.inference_chunk or FULL_CHUNK,
            context=model.context,
        )
        predicted = logits.argmax(dim=0).numpy().astype(np.uint8)[cut:]

        confusion += confusion_matrix(y, predicted, 3)
        found = gesture_events(predicted, 1, min_length, merge_gap)
        actual = segments(y == 1)
        match = match_segments(found, actual)

        for key in bites:
            bites[key] += match[key]

        true_bites += len(actual)
        predicted_bites += len(found)

    scores = per_class_scores(confusion, CLASS_NAMES)

    return {
        "eating_f1": scores["eating"]["f1"],
        "drinking_f1": scores["drinking"]["f1"],
        "bite_f1": prf(**bites)["f1"],
        "true_bites": true_bites,
        "predicted_bites": predicted_bites,
        "bite_count_error": abs(predicted_bites - true_bites),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    if args.causal and args.model != "cnn":
        raise SystemExit("--causal is only implemented for --model cnn")

    mean, std = load_normalization(NORMALIZATION_PATH)
    store = RecordingStore(CACHE_DIR)
    all_rows = read_all_windows()

    by_participant = defaultdict(list)
    for row in all_rows:
        by_participant[int(row["participant_id"])].append(row)

    folds, folds_source = load_folds(args, by_participant)
    fold_indices = args.folds if args.folds else list(range(len(folds)))
    budgets = sorted(args.budgets)
    longest = round(max(budgets) * 60 * SAMPLE_RATE)

    run_dir = make_run_dir(f"fd1_personalize_{args.model}", args.run_name)
    save_json(run_dir / "config.json", {**vars(args), "folds_source": folds_source, "folds": folds})

    print()
    print("Q5 NEW-USER EXPERIMENT")
    print("=" * 60)
    print(f"Run:     {run_dir.relative_to(ROOT)}")
    print(f"Folds:   {folds_source}, running {fold_indices}")
    print(f"Budgets: {budgets} min of the person's labeled data; modes {args.modes}")
    print()

    results = []

    for fold_index in fold_indices:
        held_out = set(folds[fold_index])
        train_rows = [
            row for row in all_rows if int(row["participant_id"]) not in held_out
        ]
        started = time.time()
        base, loss_fn = train_base(train_rows, store, mean, std, device, args)
        base.eval()
        print(f"fold {fold_index}: base model trained on {len(train_rows):,} windows "
              f"in {time.time() - started:.0f}s, held out {sorted(held_out)}")

        for participant in sorted(held_out):
            files = {}

            for row in by_participant[participant]:
                if row["file"] not in files:
                    files[row["file"]] = np.asarray(store.get(row["file"])[1])

            start = adaptation_start(files)

            if start is None or start + longest >= min(len(labels) for labels in files.values()):
                print(f"  participant {participant}: skipped (no eating, or too little data after it)")
                continue

            cut = start + longest
            entry = {
                "fold": fold_index,
                "participant_id": participant,
                "adapt_start_seconds": start / SAMPLE_RATE,
                "scored_from_seconds": cut / SAMPLE_RATE,
                "base": score_region(base, files, store, mean, std, cut, device, args),
            }

            for budget in budgets:
                rows = adaptation_windows(files, start, round(budget * 60 * SAMPLE_RATE))

                for mode in args.modes:
                    tuned = fine_tune(base, rows, store, mean, std, loss_fn, device, mode, args)
                    entry[f"{mode}_{budget:g}min"] = score_region(
                        tuned, files, store, mean, std, cut, device, args
                    )

            results.append(entry)

            line = f"  participant {participant}: base eat F1 {entry['base']['eating_f1']:.3f}"
            for budget in budgets:
                for mode in args.modes:
                    line += f" | {mode} {budget:g}m {entry[f'{mode}_{budget:g}min']['eating_f1']:.3f}"
            print(line)

    variants = ["base"] + [f"{mode}_{budget:g}min" for budget in budgets for mode in args.modes]
    averages = {}

    for variant in variants:
        scored = [entry[variant] for entry in results]

        if scored:
            averages[variant] = {
                key: float(np.mean([item[key] for item in scored]))
                for key in ("eating_f1", "drinking_f1", "bite_f1", "bite_count_error")
            }

    save_json(
        run_dir / "personalization.json",
        {"participants": len(results), "averages": averages, "per_participant": results},
    )

    print()
    print(f"AVERAGE OVER {len(results)} HELD-OUT PARTICIPANTS")
    print(f"{'variant':<14} {'eat F1':>7} {'drink F1':>9} {'bite F1':>8} {'bite count err':>15}")

    for variant, scores in averages.items():
        print(f"{variant:<14} {scores['eating_f1']:>7.3f} {scores['drinking_f1']:>9.3f} "
              f"{scores['bite_f1']:>8.3f} {scores['bite_count_error']:>15.1f}")

    print()
    print("All variants are scored on the same part of each recording: after the longest adaptation budget.")
    print(f"Saved: {(run_dir / 'personalization.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
