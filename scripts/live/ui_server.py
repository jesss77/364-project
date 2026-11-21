import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


PAGE_PATH = Path(__file__).resolve().parent / "ui.html"
WINDOW_SECONDS = 30


class LiveState:
    def __init__(self, info):
        self.lock = threading.Lock()
        self.snapshot = {"info": info, "time": 0, "finished": False}

    def publish(self, snapshot):
        with self.lock:
            self.snapshot = {**self.snapshot, **snapshot}

    def read(self):
        with self.lock:
            return json.dumps(self.snapshot).encode("utf-8")


def round_list(values, digits=3):
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def build_snapshot(stream_time, gestures, meals, counts, speed, rates):
    rate = gestures.rate
    window = WINDOW_SECONDS * rate

    samples = gestures.samples[-window:]
    samples_start = (gestures.sample_offset + len(gestures.samples) - len(samples)) / rate

    labels = gestures.labels[-window:]
    probabilities = gestures.probabilities[-window:]
    labels_start = (gestures.label_offset + len(gestures.labels) - len(labels)) / rate

    snapshot = {
        "time": stream_time,
        "rates": rates,
        "signals": {
            "start": samples_start,
            "rate": rate,
            "acc": round_list(samples[:, :3], 2),
            "gyro": round_list(samples[:, 3:], 1),
        },
        "model": {
            "start": labels_start,
            "rate": rate,
            "labels": labels.tolist(),
            "eating": round_list(probabilities[:, 1], 3) if len(probabilities) else [],
            "drinking": round_list(probabilities[:, 2], 3) if len(probabilities) else [],
        },
        "events": gestures.events[-40:],
        "counts": counts,
        "speed": speed,
        "meal": None,
    }

    if meals is not None:
        snapshot["meal"] = {
            "in_meal": meals.in_meal,
            "since": meals.meal_start,
            "history": meals.history[-40:],
            "start_threshold": meals.start_threshold,
            "end_threshold": meals.end_threshold,
            "segment_seconds": meals.stride / meals.rate,
        }

    return snapshot


def start_ui(port, state):
    page = PAGE_PATH.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state"):
                body = state.read()
                content_type = "application/json"
            elif self.path in ("/", "/index.html"):
                body = page
                content_type = "text/html; charset=utf-8"
            else:
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Dashboard: http://localhost:{port}/")
    return server
