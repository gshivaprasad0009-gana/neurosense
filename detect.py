"""
Detect — score real readings with the trained model.

Point it at an ESP32 Environment Monitor database and it reports which windows
look anomalous, with their timestamps.

Usage:
    SENSOR_DB=../esp32-environment-monitor/data/sensors.db python detect.py
    SENSOR_DB=... python detect.py --percentile 99.5
"""

import argparse
import os
import sqlite3

import numpy as np
import torch

import config
import data
import model as model_module


def load_with_time(path: str):
    """Readings plus their timestamps, so anomalies can be reported in time."""
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT ts, temperature, humidity, COALESCE(light, 50.0) "
        "FROM readings ORDER BY id"
    ).fetchall()
    conn.close()
    times = [r[0] for r in rows]
    values = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
    return times, values


def main():
    parser = argparse.ArgumentParser(description="Score sensor readings for anomalies.")
    parser.add_argument("--percentile", type=float, default=config.THRESHOLD_PERCENTILE,
                        help="Alert threshold, as a percentile of normal reconstruction error.")
    parser.add_argument("--db", default=config.SENSOR_DB, help="Path to the sensor database.")
    args = parser.parse_args()

    if not os.path.exists(config.MODEL_PATH):
        raise SystemExit("No trained model. Run `python train.py` first.")
    if not args.db or not os.path.exists(args.db):
        raise SystemExit(
            "No sensor database. Set SENSOR_DB to an esp32-environment-monitor "
            "database, or pass --db."
        )

    times, values = load_with_time(args.db)
    scaler = data.Scaler.load(config.SCALER_PATH)
    model = model_module.load_model(config.MODEL_PATH)

    windows, _ = data.windowise(scaler.transform(values))
    errors = model.reconstruction_error(torch.from_numpy(windows.astype(np.float32))).numpy()

    # With no labels to learn from, the honest question is "show me the most
    # unusual N% of windows" — so the threshold is a percentile of the scores
    # themselves. At the default 99, roughly 1% of windows are flagged.
    threshold = float(np.percentile(errors, args.percentile))

    flagged = np.where(errors > threshold)[0]
    print(f"Readings         : {len(values)}")
    print(f"Windows scored   : {len(windows)}")
    print(f"Threshold        : {threshold:.5f}  ({args.percentile}th percentile of all windows)")
    print(f"Anomalous windows: {len(flagged)}  ({100 * len(flagged) / len(windows):.1f}%)")
    print()
    if len(flagged) == 0:
        print("Nothing unusual found — the readings look normal.")
        return

    print(f"  {'when (last reading of window)':<26}{'score':>9}  readings")
    print("  " + "-" * 62)
    for index in flagged[:25]:
        end = index + config.WINDOW - 1
        window = values[index:index + config.WINDOW]
        print(f"  {str(times[end]):<26}{errors[index]:>9.4f}  "
              f"temp {window[:, 0].min():.1f}–{window[:, 0].max():.1f} °C")
    if len(flagged) > 25:
        print(f"  … and {len(flagged) - 25} more")


if __name__ == "__main__":
    main()
