"""
Data pipeline.

Two sources of data:

* `make_synthetic()` — a **labelled benchmark**. Real sensor data has no
  ground truth ("was this spike an anomaly?" is a judgement call), so to
  measure the model honestly we build a series where we *know* which readings
  are anomalous, because we put them there. The anomaly types are the ones a
  real lab actually sees: spikes, level shifts, and stuck sensors.

* `load_from_db()` — the real thing. Reads the ESP32 Environment Monitor's
  SQLite database (same schema), for scoring live data.

Both are turned into sliding windows: the model looks at WINDOW consecutive
readings and learns to reconstruct them.
"""

import json
import os
import sqlite3

import numpy as np

import config

FEATURES = config.FEATURES


# ----------------------------------------------------------------------
# Synthetic benchmark
# ----------------------------------------------------------------------

def _normal_series(n: int, rng: np.random.Generator) -> np.ndarray:
    """A day-and-a-half of believable readings: daily curves plus noise."""
    t = np.arange(n)
    minute = t * 5                                   # five-minute sampling

    temp = 26 + 3 * np.sin(2 * np.pi * (minute / 1440) - np.pi / 2)
    humidity = 60 - 6 * np.sin(2 * np.pi * (minute / 1440) - np.pi / 2)
    light = np.clip(50 + 45 * np.sin(2 * np.pi * (minute / 1440) - np.pi / 2), 0, 100)

    series = np.column_stack([temp, humidity, light])
    series += rng.normal(0, [0.5, 1.5, 2.0], size=series.shape)
    return series


def _inject_anomalies(series: np.ndarray, rate: float, rng: np.random.Generator):
    """Damage the series in three realistic ways, and record where.

    `rate` is the fraction of **readings** that end up anomalous, which is the
    number that matters — an anomaly lasting six readings is six times the
    problem of one lasting a single reading.
    """
    n = len(series)
    labels = np.zeros(n, dtype=int)
    target = max(1, int(n * rate))
    written = 0
    kind = 0
    attempts = 0

    while written < target and attempts < n:
        attempts += 1
        start = int(rng.integers(0, max(1, n - 30)))
        length = int(rng.integers(3, 10))
        end = min(start + length, n)
        if labels[start:end].any():
            continue                      # keep the anomaly segments separate

        labels[start:end] = 1
        written += end - start

        if kind % 3 == 0:                 # a spike — briefly reads far too high
            series[start:end, 0] += rng.uniform(18, 28)
        elif kind % 3 == 1:               # a level shift — drifts and stays wrong
            series[start:end, 0] += rng.uniform(8, 12)
            series[start:end, 1] -= rng.uniform(10, 15)
        else:                             # a stuck sensor — readings freeze
            series[start:end] = series[start]
        kind += 1

    return series, labels


def make_synthetic(n: int | None = None, anomaly_rate: float | None = None, seed: int | None = None):
    """Build a labelled benchmark series. Returns (readings, point_labels)."""
    n = n or config.N_NORMAL
    anomaly_rate = anomaly_rate if anomaly_rate is not None else config.ANOMALY_RATE
    seed = seed if seed is not None else config.SEED

    rng = np.random.default_rng(seed)
    series = _normal_series(n, rng)
    series, labels = _inject_anomalies(series, anomaly_rate, rng)
    return series, labels


def windowise(series: np.ndarray, labels: np.ndarray | None = None, window: int | None = None):
    """Turn a series into overlapping windows.

    A window is labelled anomalous if *any* reading inside it is anomalous —
    the model sees a window, so the window is what has to be judged.
    """
    window = window or config.WINDOW
    xs = np.stack([series[i:i + window] for i in range(len(series) - window + 1)])
    if labels is None:
        return xs, None
    ys = np.array([
        int(labels[i:i + window].any()) for i in range(len(series) - window + 1)
    ])
    return xs, ys


# ----------------------------------------------------------------------
# Real data
# ----------------------------------------------------------------------

def load_from_db(path: str | None = None) -> np.ndarray:
    """Read readings from an ESP32 Environment Monitor database."""
    path = path or config.SENSOR_DB
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "No sensor database. Either set SENSOR_DB to an esp32-environment-monitor "
            "database, or use the synthetic benchmark (the default)."
        )
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT temperature, humidity, COALESCE(light, 50.0) FROM readings ORDER BY id"
    ).fetchall()
    conn.close()
    if len(rows) < config.WINDOW:
        raise ValueError(f"Need at least {config.WINDOW} readings, found {len(rows)}")
    return np.array(rows, dtype=float)


# ----------------------------------------------------------------------
# Scaling and splitting
# ----------------------------------------------------------------------

class Scaler:
    """Standardise features. Saved to JSON so scoring new data uses the same
    numbers the model was trained with — a classic source of silent bugs."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, series: np.ndarray) -> "Scaler":
        self.mean = series.mean(axis=0)
        self.std = series.std(axis=0) + 1e-8
        return self

    def transform(self, series: np.ndarray) -> np.ndarray:
        return (series - self.mean) / self.std

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)

    @classmethod
    def load(cls, path: str) -> "Scaler":
        with open(path) as f:
            data = json.load(f)
        return cls(np.array(data["mean"]), np.array(data["std"]))


def prepare_benchmark(seed: int | None = None):
    """Build the benchmark and split it into train / validation / test.

    The split is by *time*, not at random: training on the future to detect the
    past would flatter the model. And the scaler is fitted on the training
    portion only, so no information leaks across the boundary.
    """
    seed = seed if seed is not None else config.SEED
    series, labels = make_synthetic(seed=seed)

    n = len(series)
    train_end = int(n * 0.5)
    val_end = int(n * 0.75)

    scaler = Scaler().fit(series[:train_end])
    scaled = scaler.transform(series)

    splits = {}
    for name, (a, b) in {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n),
    }.items():
        xs, ys = windowise(scaled[a:b], labels[a:b])
        splits[name] = (xs.astype(np.float32), ys.astype(int))
    return splits, scaler, (series, labels)
