import argparse
import copy
import io
import warnings
from argparse import Namespace

import numpy as np
import torch
from torch.ao.quantization import get_default_qconfig_mapping, quantize_dynamic
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from common import (
    ROOT,
    RUNS_DIR,
    RecordingStore,
    WindowDataset,
    load_json,
    load_normalization,
    load_windows,
    save_json,
    set_seed,
)
from models import build_gesture_model
from train_fd1 import (
    CACHE_DIR,
    NORMALIZATION_PATH,
    WINDOWS_PATH,
    cpu_ms_per_minute,
    evaluate,
    gesture_target,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a trained KU Leuven gesture model to int8 and compare it with float32."
    )
    parser.add_argument("run", help="Run name under results/runs")
    parser.add_argument("--calibration-windows", type=int, default=256)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pick_engine():
    engines = torch.backends.quantized.supported_engines

    for engine in ("qnnpack", "x86", "fbgemm"):
        if engine in engines:
            torch.backends.quantized.engine = engine
            return engine

    raise SystemExit(f"No int8 engine available (found {engines}).")


def serialized_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / 1e6


def load_model(run_dir, config):
    options = {"channels": config["channels"], "dropout": config["dropout"]}

    if config.get("causal"):
        options["causal"] = True

    model = build_gesture_model(config["model"], **options)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    return model.eval()


def calibration_batches(rows, store, mean, std, count, seed):
    rng = np.random.default_rng(seed)
    chosen = [rows[index] for index in rng.choice(len(rows), size=min(count, len(rows)), replace=False)]
    dataset = WindowDataset(chosen, store, mean, std, gesture_target)

    for start in range(0, len(dataset), 32):
        yield torch.stack([dataset[index][0] for index in range(start, min(start + 32, len(dataset)))])


def quantize(model, batches, engine):
    example = next(iter(batches))

    try:
        prepared = prepare_fx(copy.deepcopy(model), get_default_qconfig_mapping(engine), (example,))

        with torch.no_grad():
            prepared(example)
            for batch in batches:
                prepared(batch)

        quantized = convert_fx(prepared)
        method = "static int8 (FX graph mode, calibrated on training windows)"
    except Exception as error:
        quantized = quantize_dynamic(
            copy.deepcopy(model), {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8
        )
        method = f"dynamic int8 on Linear/GRU layers (static failed: {type(error).__name__})"

    for name in ("context", "lookahead", "inference_chunk", "receptive_field"):
        if hasattr(model, name):
            setattr(quantized, name, getattr(model, name))

    return quantized.eval(), method


def summary(metrics):
    return {
        "eating_f1": metrics["sample"]["per_class"]["eating"]["f1"],
        "drinking_f1": metrics["sample"]["per_class"]["drinking"]["f1"],
        "bite_event_f1": metrics["events"]["eating"]["f1"],
        "bite_count_mae": metrics["bite_count"]["per_participant_error"]["mae"],
    }


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    set_seed(args.seed)

    run_dir = RUNS_DIR / args.run
    config = load_json(run_dir / "config.json")
    engine = pick_engine()
    torch.backends.mha.set_fastpath_enabled(False)

    mean, std = load_normalization(NORMALIZATION_PATH)
    store = RecordingStore(CACHE_DIR)
    rows = {
        split_name: load_windows(WINDOWS_PATH, split_name)
        for split_name in ("train", "validation", "test")
    }

    model = load_model(run_dir, config)
    batches = list(
        calibration_batches(rows["train"], store, mean, std, args.calibration_windows, args.seed)
    )
    quantized, method = quantize(model, batches, engine)

    scoring = Namespace(
        min_event_seconds=config["min_event_seconds"],
        merge_gap_seconds=config["merge_gap_seconds"],
        bout_gap_seconds=config.get("bout_gap_seconds", 60.0),
    )
    cpu = torch.device("cpu")
    splits = ["validation"] + (["test"] if args.eval_test else [])

    result = {
        "run": args.run,
        "model": config["model"] + ("_causal" if config.get("causal") else ""),
        "engine": engine,
        "method": method,
        "calibration_windows": sum(len(batch) for batch in batches),
        "float32": {
            "size_mb": serialized_mb(model),
            "cpu_ms_per_minute_single_thread": cpu_ms_per_minute(model),
        },
        "int8": {
            "size_mb": serialized_mb(quantized),
            "cpu_ms_per_minute_single_thread": cpu_ms_per_minute(quantized),
        },
    }

    for split_name in splits:
        result["float32"][split_name] = summary(
            evaluate(model, rows[split_name], store, mean, std, cpu, scoring)
        )
        result["int8"][split_name] = summary(
            evaluate(quantized, rows[split_name], store, mean, std, cpu, scoring)
        )

    output_path = run_dir / "quantization.json"
    save_json(output_path, result)

    print(f"INT8 CONVERSION  run {args.run}  model {result['model']}")
    print(f"Engine: {engine}")
    print(f"Method: {method}")
    print()
    print(f"{'':<10} {'size MB':>8} {'ms/min':>8}", end="")
    for split_name in splits:
        print(f"  {split_name[:4]} eat F1  bite F1  bite MAE", end="")
    print()

    for precision in ("float32", "int8"):
        entry = result[precision]
        print(f"{precision:<10} {entry['size_mb']:>8.3f} {entry['cpu_ms_per_minute_single_thread']:>8.1f}", end="")
        for split_name in splits:
            scores = entry[split_name]
            print(f"  {scores['eating_f1']:>11.3f} {scores['bite_event_f1']:>8.3f} {scores['bite_count_mae']:>9.1f}", end="")
        print()

    print()
    print("CPU time is measured on this computer. int8 speed-ups mostly show on phone and watch chips.")
    print(f"Saved: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
