import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from common import ROOT
from convert import RAD_TO_DEG, load_profiles


PARTICIPANT_DIR = ROOT / "data" / "processed" / "kuleuven_fd1" / "participants"
SOURCE_RATE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write a phone-like CSV from a KU Leuven recording, with its true labels, "
            "to test the live pipeline before recording a real phone."
        )
    )
    parser.add_argument("--participant", type=int, required=True)
    parser.add_argument("--wrist", choices=["L", "R"], default="R")
    parser.add_argument("--start-minutes", type=float, default=None,
                        help="Default: 2 minutes before the first bite")
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--rate", type=float, default=50, help="Readings per second in the output")
    parser.add_argument("--profile", default="android", help="Source profile to imitate")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    matches = list(PARTICIPANT_DIR.glob(f"*/participant_{args.participant:02d}_{args.wrist}.npz"))

    if not matches:
        raise SystemExit(f"No processed file for participant {args.participant} wrist {args.wrist}")

    with np.load(matches[0], allow_pickle=False) as data:
        X = data["X"].astype(np.float64)
        y = data["y"]

    if args.start_minutes is None:
        eating = np.flatnonzero(y == 1)
        start = max(int(eating[0]) - 2 * 60 * SOURCE_RATE, 0) if len(eating) else 0
    else:
        start = round(args.start_minutes * 60 * SOURCE_RATE)

    end = min(start + round(args.minutes * 60 * SOURCE_RATE), len(y))
    X = X[start:end]
    y = y[start:end]

    source_times = np.arange(len(y)) / SOURCE_RATE
    out_times = np.arange(0, source_times[-1], 1 / args.rate)
    columns = np.column_stack(
        [np.interp(out_times, source_times, X[:, channel]) for channel in range(6)]
    )
    labels = y[np.minimum(np.round(out_times * SOURCE_RATE).astype(int), len(y) - 1)]

    profile = load_profiles()["sources"][args.profile]

    if profile["gravity"] != "included":
        raise SystemExit("Replay files keep gravity in the accelerometer; pick a profile with gravity 'included'")

    if profile["acc_unit"] == "g":
        columns[:, :3] /= 9.80665
    if profile["gyro_unit"] == "rad/s":
        columns[:, 3:] /= RAD_TO_DEG

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = "time_s,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,label"
    table = np.column_stack([out_times, columns, labels])
    np.savetxt(output, table, delimiter=",", header=header, comments="",
               fmt=["%.4f"] + ["%.6f"] * 6 + ["%d"])

    bites = int(np.sum(np.diff(np.concatenate(([0], (y == 1).astype(int)))) == 1))
    print(f"Wrote {output} ({len(out_times):,} rows, {args.rate:g}/s, profile {args.profile})")
    print(f"From participant {args.participant} {args.wrist}, "
          f"{start / SOURCE_RATE / 60:.1f}-{end / SOURCE_RATE / 60:.1f} min, {bites} true bites")


if __name__ == "__main__":
    main()
