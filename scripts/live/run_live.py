import argparse
import json
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from common import ROOT, RUNS_DIR, load_json, load_normalization, normalize
from convert import StreamConverter, load_profiles, read_any_recording
from metrics import clean_segments, segments
from models import EpisodeCNN, build_gesture_model
from postprocess import eating_speed
from ui_server import LiveState, build_snapshot, start_ui


EVENT_LABELS = {1: "bite", 2: "sip"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the trained models on a live or replayed wrist stream.")
    parser.add_argument("--gesture-run", required=True, help="KU Leuven run under results/runs")
    parser.add_argument("--meal-run", help="Clemson run under results/runs (optional)")
    parser.add_argument("--profile", default="sensor_logger", help="Source profile in config/live_profiles.json")
    parser.add_argument("--save", metavar="FILE.jsonl", help="With --listen: also save every pushed message, to replay later")
    parser.add_argument("--axes", default="x,y,z", help="Phone axis to training axis map, e.g. '-y,x,z'")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--replay",
        help="Recording to play back: our CSV, a Sensor Logger export (.zip or folder), or pushed messages saved with --save (.jsonl)",
    )
    source.add_argument("--listen", type=int, metavar="PORT", help="Accept JSON readings over HTTP POST")
    parser.add_argument("--speed", type=float, default=0, help="Replay speed (1 = real time, 0 = as fast as possible)")
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--ui", type=int, metavar="PORT", help="Serve a live dashboard on http://localhost:PORT/")
    parser.add_argument(
        "--show-stream",
        type=float,
        default=0,
        metavar="SECONDS",
        help="Print a line about the incoming stream every N seconds (0 = off)",
    )
    return parser.parse_args()


def run_normalization(config, default_path):
    stored = config.get("normalization")

    if stored:
        return (
            np.asarray(stored["mean"], dtype=np.float32),
            np.asarray(stored["std"], dtype=np.float32),
        )

    return load_normalization(default_path)


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class GestureTracker:
    def __init__(self, run_name):
        run_dir = RUNS_DIR / run_name
        config = load_json(run_dir / "config.json")
        options = {"channels": config["channels"], "dropout": config["dropout"]}

        if config.get("causal"):
            options["causal"] = True

        self.model = build_gesture_model(config["model"], **options)
        self.model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
        self.model.eval()

        self.rate = 16
        self.mean, self.std = run_normalization(config, ROOT / "config" / "fd1_normalization.json")
        self.min_length = round(config["min_event_seconds"] * self.rate)
        self.merge_gap = round(config["merge_gap_seconds"] * self.rate)
        self.context = self.model.context
        self.lookahead = self.model.lookahead
        self.keep = self.context + self.lookahead + 120 * self.rate

        self.samples = np.zeros((0, 6), dtype=np.float32)
        self.sample_offset = 0
        self.labels = np.zeros(0, dtype=np.uint8)
        self.probabilities = np.zeros((0, 3), dtype=np.float32)
        self.label_offset = 0
        self.finalized = 0
        self.emitted_until = {label: -1 for label in EVENT_LABELS}
        self.events = []

        print(f"Gesture model: {run_name} ({config['model']}{', causal' if config.get('causal') else ''}), "
              f"look-ahead {self.lookahead / self.rate:.1f} s")

    @torch.no_grad()
    def update(self, new_samples):
        if not len(new_samples):
            return []

        self.samples = np.vstack([self.samples, new_samples])
        now = self.sample_offset + len(self.samples)
        ready = now - self.lookahead

        if ready <= self.finalized:
            return []

        left = max(self.finalized - self.context, self.sample_offset)
        piece = normalize(self.samples[left - self.sample_offset:], self.mean, self.std)
        logits = self.model(torch.from_numpy(piece.T.copy()).unsqueeze(0))[0]
        window = slice(self.finalized - left, ready - left)
        new_labels = logits.argmax(dim=0).numpy().astype(np.uint8)[window]
        new_probabilities = torch.softmax(logits, dim=0).numpy().T[window].astype(np.float32)

        self.labels = np.concatenate([self.labels, new_labels])
        self.probabilities = np.vstack([self.probabilities, new_probabilities])
        self.finalized = ready

        found = []
        closed_before = self.finalized - self.merge_gap - 1

        for label, name in EVENT_LABELS.items():
            for start, end in clean_segments(segments(self.labels == label), self.merge_gap, self.min_length):
                start += self.label_offset
                end += self.label_offset

                if end <= closed_before and start > self.emitted_until[label]:
                    event = {"type": name, "start": start / self.rate, "end": end / self.rate}
                    self.events.append(event)
                    found.append(event)
                    self.emitted_until[label] = end

        self._trim()
        return sorted(found, key=lambda event: event["start"])

    def _trim(self):
        extra = len(self.samples) - self.keep
        if extra > 0:
            self.samples = self.samples[extra:]
            self.sample_offset += extra

        extra = len(self.labels) - 300 * self.rate
        if extra > 0:
            self.labels = self.labels[extra:]
            self.probabilities = self.probabilities[extra:]
            self.label_offset += extra


class MealTracker:
    def __init__(self, run_name):
        run_dir = RUNS_DIR / run_name
        config = load_json(run_dir / "config.json")
        metrics = load_json(run_dir / "metrics.json")["validation"]

        self.model = EpisodeCNN(dropout=config["dropout"])
        self.model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
        self.model.eval()

        self.rate = 15
        self.mean, self.std = run_normalization(config, ROOT / "config" / "clemson_normalization.json")
        self.window = config["window_samples"]
        self.stride = config["stride_samples"]
        self.start_threshold = metrics["start_threshold"]
        self.end_threshold = metrics["end_threshold"]

        self.samples = np.zeros((0, 6), dtype=np.float32)
        self.sample_offset = 0
        self.window_index = 0
        self.previous_probability = None
        self.in_meal = False
        self.meal_start = None
        self.history = []

        print(f"Meal model: {run_name}, thresholds start {self.start_threshold} / end {self.end_threshold}, "
              f"updates every {self.stride / self.rate / 60:.0f} min")

    @torch.no_grad()
    def update(self, new_samples):
        self.samples = np.vstack([self.samples, new_samples])
        changes = []

        while True:
            start = self.window_index * self.stride
            end = start + self.window

            if self.sample_offset + len(self.samples) < end:
                break

            piece = normalize(self.samples[start - self.sample_offset:end - self.sample_offset], self.mean, self.std)
            probability = torch.sigmoid(
                self.model(torch.from_numpy(piece.T.copy()).unsqueeze(0))
            ).item()

            segment_probability = (
                probability
                if self.previous_probability is None
                else (probability + self.previous_probability) / 2
            )
            segment_start = start / self.rate
            self.history.append({"start": segment_start, "probability": segment_probability})

            if not self.in_meal and segment_probability >= self.start_threshold:
                self.in_meal = True
                self.meal_start = segment_start
                changes.append({"type": "meal started", "time": segment_start, "probability": segment_probability})
            elif self.in_meal and segment_probability < self.end_threshold:
                self.in_meal = False
                changes.append({"type": "meal ended", "time": segment_start, "start": self.meal_start,
                                "probability": segment_probability})

            self.previous_probability = probability
            self.window_index += 1

            drop = (self.window_index * self.stride) - self.sample_offset
            if drop > 0:
                self.samples = self.samples[drop:]
                self.sample_offset += drop

        return changes


def meal_summary(events, start, end):
    bites = [e for e in events if e["type"] == "bite" and start <= e["start"] < end]
    sips = [e for e in events if e["type"] == "sip" and start <= e["start"] < end]
    minutes = (end - start) / 60
    rate = len(bites) / minutes if minutes else 0
    return f"{len(bites)} bites, {len(sips)} sips, {minutes:.0f} min, {rate:.1f} bites/min"


def timed_chunks(times, acc, gyro, gravity, speed, chunk_seconds=0.5):
    started = time.time()
    index = 0

    while index < len(times):
        stop = max(int(np.searchsorted(times, times[index] + chunk_seconds)), index + 1)

        if speed > 0:
            wait = (times[index] - times[0]) / speed - (time.time() - started)
            if wait > 0:
                time.sleep(wait)

        yield (
            times[index:stop],
            acc[index:stop],
            gyro[index:stop],
            None if gravity is None else gravity[index:stop],
        )
        index = stop


def replay_source(path, speed):
    if str(path).endswith(".jsonl"):
        yield from pushed_messages_source(path, speed)
        return

    times, acc, gyro, gravity, _ = read_any_recording(path)
    yield from timed_chunks(times, acc, gyro, gravity, speed)


SENSOR_NAMES = {"accelerometer": "acc", "gyroscope": "gyro", "gravity": "gravity"}


def entry_vector(entry):
    values = entry.get("values", entry)
    vector = [values.get("x"), values.get("y"), values.get("z")]
    return None if None in vector else [float(value) for value in vector]


def parse_payload(data, state):
    if isinstance(data, dict) and "time_s" in data:
        return (
            np.asarray(data["time_s"], dtype=np.float64),
            np.asarray(data["acc"], dtype=np.float64),
            np.asarray(data["gyro"], dtype=np.float64),
            np.asarray(data["gravity"], dtype=np.float64) if "gravity" in data else None,
        )

    entries = data.get("payload", []) if isinstance(data, dict) else data

    if not entries:
        print("Test message received (no sensor data). The phone can reach this computer.")
        return None

    names = {str(entry.get("name", "")).lower() for entry in entries}
    state.setdefault("names", set()).update(names)

    if not state.get("announced"):
        print(f"First data from the phone. Sensors in this message: {', '.join(sorted(names))}")
        state["announced"] = True

    for needed in ("accelerometer", "gyroscope", "gravity"):
        if needed not in state["names"] and not state.get(f"warned_{needed}") and len(state.get("messages", [])) > 3:
            print(f"No '{needed}' readings yet: turn on {needed.capitalize()} in Sensor Logger.")
            state[f"warned_{needed}"] = True

    state.setdefault("messages", []).append(1)
    rows = []

    for entry in sorted(entries, key=lambda item: item.get("time", 0)):
        kind = SENSOR_NAMES.get(str(entry.get("name", "")).lower())
        vector = entry_vector(entry)

        if kind is None or vector is None:
            continue

        stamp = float(entry.get("time", 0))
        stamp = stamp / 1e9 if stamp > 1e12 else stamp

        if kind != "acc":
            state[kind] = vector
            continue

        if state.get("gyro") is None or state.get("gravity") is None:
            continue

        if stamp <= state.get("last_time", -np.inf):
            continue

        state["last_time"] = stamp
        rows.append((stamp, vector, state["gyro"], state["gravity"]))

    if not rows:
        return None

    times = np.array([row[0] for row in rows])

    if state.get("first_time") is None:
        state["first_time"] = times[0]

    return (
        times - state["first_time"],
        np.array([row[1] for row in rows]),
        np.array([row[2] for row in rows]),
        np.array([row[3] for row in rows]),
    )


def local_address():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return "<this computer's IP>"
    finally:
        probe.close()


def listen_source(port, save_path=None):
    readings = queue.Queue()
    state = {}
    save_file = open(save_path, "a", encoding="utf-8") if save_path else None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            print(f"Browser visit from {self.client_address[0]}: the phone can reach this computer.")
            body = b"Connected to the eating-detection listener.\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))

            try:
                data = json.loads(body)

                if save_file is not None:
                    save_file.write(json.dumps(data) + "\n")
                    save_file.flush()

                parsed = parse_payload(data, state)

                if parsed is not None:
                    readings.put(parsed)

                self.send_response(200)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                print(f"Could not read a message: {error}")
                self.send_response(400)

            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Listening. In Sensor Logger set the HTTP Push URL to: http://{local_address()}:{port}/")

    try:
        while True:
            yield readings.get()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        if save_file is not None:
            save_file.close()


def pushed_messages_source(path, speed):
    state = {}
    chunks = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parsed = parse_payload(json.loads(line), state)
            if parsed is not None:
                chunks.append(parsed)

    if not chunks:
        raise SystemExit(f"{path}: no accelerometer readings with gyroscope and gravity")

    times = np.concatenate([chunk[0] for chunk in chunks])
    acc = np.vstack([chunk[1] for chunk in chunks])
    gyro = np.vstack([chunk[2] for chunk in chunks])
    gravity = np.vstack([chunk[3] for chunk in chunks])
    yield from timed_chunks(times, acc, gyro, gravity, speed)


def main():
    args = parse_args()
    profiles = load_profiles()
    source_profile = profiles["sources"][args.profile]

    gestures = GestureTracker(args.gesture_run)
    gesture_converter = StreamConverter(source_profile, profiles["targets"]["fd1"], args.axes)

    meals = MealTracker(args.meal_run) if args.meal_run else None
    meal_converter = (
        StreamConverter(source_profile, profiles["targets"]["clemson"], args.axes) if meals else None
    )

    state = None
    if args.ui:
        state = LiveState(
            {
                "source": f"replay {Path(args.replay).name}" if args.replay else f"listening on port {args.listen}",
                "gesture_model": args.gesture_run,
                "meal_model": args.meal_run,
                "lookahead": gestures.lookahead / gestures.rate,
                "profile": args.profile,
            }
        )
        start_ui(args.ui, state)

    print()
    source = replay_source(args.replay, args.speed) if args.replay else listen_source(args.listen, args.save)
    pending = np.zeros((0, 6), dtype=np.float32)
    step = round(args.step_seconds * gestures.rate)
    stream_end = 0.0
    counts = {"bite": 0, "sip": 0}

    recent = []
    raw_count = 0
    next_status = args.show_stream
    rate_window = []
    last_publish = 0.0

    def publish(finished=False):
        rate_window_span = (rate_window[-1][0] - rate_window[0][0]) if len(rate_window) > 1 else 0
        rates = None
        if rate_window_span > 0:
            rates = {
                "phone": sum(item[1] for item in rate_window[1:]) / rate_window_span,
                "model": sum(item[2] for item in rate_window[1:]) / rate_window_span,
            }
        bites = [(round(e["start"] * 16), round(e["end"] * 16)) for e in gestures.events if e["type"] == "bite"]
        snapshot = build_snapshot(stream_end, gestures, meals, dict(counts), eating_speed(bites, 16), rates)
        snapshot["events"] = sorted(snapshot["events"] + meal_events[-10:], key=lambda e: e["start"])[-40:]
        snapshot["finished"] = finished
        state.publish(snapshot)

    meal_events = []

    try:
        for times, acc, gyro, gravity in source:
            stream_end = float(times[-1])
            converted = gesture_converter.push(times, acc, gyro, gravity)
            pending = np.vstack([pending, converted])
            rate_window.append((stream_end, len(times), len(converted)))
            rate_window = [item for item in rate_window if item[0] >= stream_end - 5]

            if args.show_stream:
                recent.append(converted)
                raw_count += len(times)

                if stream_end >= next_status:
                    block = np.vstack(recent) if recent else np.zeros((0, 6))
                    if len(block):
                        acc_mean = block[:, :3].mean(axis=0)
                        print(
                            f"[{clock(stream_end)}] stream: phone {raw_count / args.show_stream:.0f}/s "
                            f"→ model {len(block) / args.show_stream:.0f}/s | "
                            f"acc mean ({acc_mean[0]:+.1f}, {acc_mean[1]:+.1f}, {acc_mean[2]:+.1f}) "
                            f"|acc| {np.linalg.norm(block[:, :3], axis=1).mean():.1f} m/s² | "
                            f"gyro peak {np.abs(block[:, 3:]).max():.0f} °/s"
                        )
                    recent = []
                    raw_count = 0
                    next_status = stream_end + args.show_stream

            if len(pending) >= step:
                for event in gestures.update(pending):
                    counts[event["type"]] += 1
                    print(f"[{clock(event['start'])}] {event['type']} {counts[event['type']]}")
                pending = pending[:0]

            if meals:
                for change in meals.update(meal_converter.push(times, acc, gyro, gravity)):
                    if change["type"] == "meal started":
                        text = "meal started"
                    else:
                        text = f"meal ended: {meal_summary(gestures.events, change['start'], change['time'])}"
                    print(f"[{clock(change['time'])}] {text}")
                    meal_events.append({"type": "meal", "start": change["time"], "end": change["time"], "text": text})

            if state is not None and time.time() - last_publish >= 0.2:
                publish()
                last_publish = time.time()
    except KeyboardInterrupt:
        pass

    if len(pending):
        gestures.update(pending)

    print()
    print(f"Stream length {clock(stream_end)}: {counts['bite']} bites, {counts['sip']} sips")

    bites = [(round(e["start"] * 16), round(e["end"] * 16)) for e in gestures.events if e["type"] == "bite"]
    speed = eating_speed(bites, 16)
    if speed["bites_per_minute"]:
        print(f"Eating speed: {speed['bites_per_minute']:.1f} bites/min, "
              f"median {speed['median_seconds_between_bites']:.1f} s between bites")

    if args.replay and not str(args.replay).endswith(".jsonl"):
        labels = read_any_recording(args.replay)[4]
        if labels is not None:
            true_bites = int(np.sum(np.diff(np.concatenate(([0], (labels == 1).astype(int)))) == 1))
            true_sips = int(np.sum(np.diff(np.concatenate(([0], (labels == 2).astype(int)))) == 1))
            print(f"Labels in the replay file: {true_bites} bites, {true_sips} sips")

    if state is not None:
        publish(finished=True)
        print()
        print(f"Dashboard stays open at http://localhost:{args.ui}/ until you press Ctrl-C.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
