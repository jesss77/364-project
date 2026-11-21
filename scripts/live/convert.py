import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = ROOT / "config" / "live_profiles.json"

STANDARD_GRAVITY = 9.80665
RAD_TO_DEG = 180.0 / np.pi

GRAVITY_TIME_CONSTANT_SECONDS = 1.0


def load_profiles(path=PROFILES_PATH):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def acc_to_mps2(values, unit):
    return values * STANDARD_GRAVITY if unit == "g" else values


def mps2_to_acc(values, unit):
    return values / STANDARD_GRAVITY if unit == "g" else values


def gyro_to_degps(values, unit):
    return values * RAD_TO_DEG if unit == "rad/s" else values


def degps_to_gyro(values, unit):
    return values / RAD_TO_DEG if unit == "rad/s" else values


def parse_axes(spec):
    parts = [part.strip() for part in spec.split(",")]

    if sorted(part.lstrip("-") for part in parts) != ["x", "y", "z"]:
        raise ValueError(f"Axis map must use x, y and z once each, got {spec!r}")

    index = {"x": 0, "y": 1, "z": 2}
    matrix = np.zeros((3, 3))

    for row, part in enumerate(parts):
        matrix[row, index[part.lstrip("-")]] = -1.0 if part.startswith("-") else 1.0

    return matrix


class StreamConverter:
    def __init__(self, source, target, axes="x,y,z"):
        self.source = source
        self.target = target
        self.rate = float(target["rate_hz"])
        self.axes = parse_axes(axes)

        self.gravity_estimate = None
        self.last_time = None

        self.next_output_time = None
        self.bin_sum = np.zeros(6)
        self.bin_count = 0
        self.last_output = None

    def _gravity_handling(self, times, acc, gravity):
        mode = self.source["gravity"]
        wanted = self.target["gravity"]

        if mode == "separate":
            if gravity is None:
                raise ValueError("Source profile has separate gravity but no gravity values were given")
            return acc + gravity if wanted == "include" else acc

        if mode == "removed":
            if wanted == "include":
                raise ValueError("Cannot add gravity back: the source has it removed and gives no gravity vector")
            return acc

        if wanted == "include":
            return acc

        output = np.empty_like(acc)

        for index, (time, sample) in enumerate(zip(times, acc)):
            if self.gravity_estimate is None:
                self.gravity_estimate = sample.copy()
            else:
                step = max(time - self.last_time, 1e-3)
                alpha = step / (GRAVITY_TIME_CONSTANT_SECONDS + step)
                self.gravity_estimate += alpha * (sample - self.gravity_estimate)

            self.last_time = time
            output[index] = sample - self.gravity_estimate

        return output

    def push(self, times, acc, gyro, gravity=None):
        times = np.asarray(times, dtype=np.float64)
        acc = acc_to_mps2(np.asarray(acc, dtype=np.float64), self.source["acc_unit"])
        gyro = gyro_to_degps(np.asarray(gyro, dtype=np.float64), self.source["gyro_unit"])

        if gravity is not None:
            gravity = acc_to_mps2(np.asarray(gravity, dtype=np.float64), self.source["acc_unit"])

        acc = self._gravity_handling(times, acc, gravity)
        acc = mps2_to_acc(acc @ self.axes.T, self.target["acc_unit"])
        gyro = degps_to_gyro(gyro @ self.axes.T, self.target["gyro_unit"])

        return self._resample(times, np.hstack([acc, gyro]))

    def _resample(self, times, values):
        period = 1.0 / self.rate
        outputs = []

        if self.next_output_time is None and len(times):
            self.next_output_time = times[0]

        for time, value in zip(times, values):
            while time >= self.next_output_time + period / 2:
                if self.bin_count:
                    sample = self.bin_sum / self.bin_count
                elif self.last_output is not None:
                    sample = self.last_output
                else:
                    sample = value

                outputs.append(sample)
                self.last_output = sample
                self.bin_sum = np.zeros(6)
                self.bin_count = 0
                self.next_output_time += period

            self.bin_sum += value
            self.bin_count += 1

        if not outputs:
            return np.zeros((0, 6), dtype=np.float32)

        return np.asarray(outputs, dtype=np.float32)


SENSOR_LOGGER_FILES = {"acc": "accelerometer.csv", "gyro": "gyroscope.csv", "gravity": "gravity.csv"}


def read_sensor_logger_table(text):
    rows = list(csv.DictReader(io.StringIO(text)))

    if not rows:
        raise ValueError("empty sensor file")

    if "seconds_elapsed" in rows[0]:
        times = np.array([float(row["seconds_elapsed"]) for row in rows])
    else:
        times = np.array([float(row["time"]) for row in rows]) / 1e9

    values = np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows])
    order = np.argsort(times, kind="stable")
    return times[order], values[order]


def read_sensor_logger_export(path):
    path = Path(path)
    texts = {}

    if path.is_dir():
        for file in path.iterdir():
            texts[file.name.lower()] = file
        read = lambda key: texts[key].read_text(encoding="utf-8")
    else:
        archive = zipfile.ZipFile(path)
        for name in archive.namelist():
            texts[Path(name).name.lower()] = name
        read = lambda key: archive.read(texts[key]).decode("utf-8")

    missing = [name for name in SENSOR_LOGGER_FILES.values() if name not in texts]

    if missing:
        raise ValueError(
            f"{path}: missing {missing}. Record with Accelerometer, Gyroscope and Gravity turned on."
        )

    tables = {key: read_sensor_logger_table(read(name)) for key, name in SENSOR_LOGGER_FILES.items()}
    times, acc = tables["acc"]

    def align(key):
        source_times, values = tables[key]
        return np.column_stack(
            [np.interp(times, source_times, values[:, axis]) for axis in range(3)]
        )

    return times - times[0], acc, align("gyro"), align("gravity"), None


def read_any_recording(path):
    path = Path(path)

    if path.is_dir() or path.suffix.lower() == ".zip":
        return read_sensor_logger_export(path)

    return read_stream_csv(path)


def read_stream_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    names = data.dtype.names

    required = ["time_s", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    missing = [name for name in required if name not in names]

    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Expected {required} (+ optional grav_x/y/z, label)")

    times = data["time_s"]
    acc = np.column_stack([data["acc_x"], data["acc_y"], data["acc_z"]])
    gyro = np.column_stack([data["gyro_x"], data["gyro_y"], data["gyro_z"]])
    gravity = (
        np.column_stack([data["grav_x"], data["grav_y"], data["grav_z"]])
        if "grav_x" in names
        else None
    )
    labels = data["label"].astype(np.int64) if "label" in names else None

    return times, acc, gyro, gravity, labels
