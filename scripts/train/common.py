import csv
import json
import random
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]

RUNS_DIR = ROOT / "results" / "runs"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_normalization(path):
    stats = load_json(path)

    return (
        np.asarray(stats["mean"], dtype=np.float32),
        np.asarray(stats["std"], dtype=np.float32),
    )


def load_windows(path, split_name):
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row["split"] == split_name
        ]


def make_run_dir(prefix, name=None):
    name = name or f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class RecordingStore:
    def __init__(self, cache_dir, max_open=64):
        self.cache_dir = Path(cache_dir)
        self.max_open = max_open
        self.open = OrderedDict()

    def _cache_paths(self, relative_file):
        source = Path(relative_file)
        base = self.cache_dir / source.parent.name / source.stem
        return base.with_name(base.name + "_X.npy"), base.with_name(base.name + "_y.npy")

    def _ensure_cache(self, relative_file):
        source = ROOT / relative_file
        x_path, y_path = self._cache_paths(relative_file)

        fresh = (
            x_path.exists()
            and y_path.exists()
            and x_path.stat().st_mtime >= source.stat().st_mtime
        )

        if not fresh:
            x_path.parent.mkdir(parents=True, exist_ok=True)

            with np.load(source, allow_pickle=False) as data:
                np.save(x_path, data["X"])
                np.save(y_path, data["y"])

        return x_path, y_path

    def get(self, relative_file):
        if relative_file in self.open:
            self.open.move_to_end(relative_file)
            return self.open[relative_file]

        x_path, y_path = self._ensure_cache(relative_file)
        arrays = (
            np.load(x_path, mmap_mode="r"),
            np.load(y_path, mmap_mode="r"),
        )

        self.open[relative_file] = arrays

        if len(self.open) > self.max_open:
            self.open.popitem(last=False)

        return arrays

    def __getstate__(self):
        state = self.__dict__.copy()
        state["open"] = OrderedDict()
        return state


def normalize(X, mean, std):
    X = (np.asarray(X, dtype=np.float32) - mean) / std
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class WindowDataset(Dataset):
    def __init__(self, rows, store, mean, std, target_fn):
        self.rows = rows
        self.store = store
        self.mean = mean
        self.std = std
        self.target_fn = target_fn

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        start = int(row["start_sample"])
        end = int(row["end_sample"])

        X, y = self.store.get(row["file"])
        window = normalize(X[start:end], self.mean, self.std)

        return (
            torch.from_numpy(window.T.copy()),
            self.target_fn(row, np.asarray(y[start:end])),
        )


@torch.no_grad()
def predict_sequence(model, X, device, chunk, context):
    model.eval()
    total = len(X)
    outputs = []

    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        left = max(start - context, 0)
        right = min(end + context, total)

        piece = torch.from_numpy(X[left:right].T.copy()).unsqueeze(0).to(device)
        logits = model(piece)[0]
        outputs.append(logits[:, start - left:end - left].float().cpu())

    return torch.cat(outputs, dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
