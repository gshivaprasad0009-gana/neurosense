"""
Evaluation — does the deep model actually beat the classical one?

Both models are scored the same way, which is the only way the comparison
means anything:

1. Fit on the normal training windows.
2. Calibrate the alert threshold on the **validation** split — never the test
   split. Tuning on test data is the most common way people accidentally
   cheat, and it makes results look better than they are.
3. Measure precision, recall and F1 on the **test** split, which neither model
   has ever seen.

Usage:
    python evaluate.py
"""

import json
import os

import numpy as np
import torch

import baseline
import config
import data
import model as model_module
from sklearn.metrics import precision_recall_fscore_support


def metrics_from_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Turn scores into a verdict and measure it against the truth."""
    predicted = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "flagged": int(predicted.sum()),
        "true_anomalies": int(labels.sum()),
        "total_windows": int(len(labels)),
    }


def calibrate(normal_scores: np.ndarray, percentile: float = None) -> float:
    """Pick the alert threshold from the error distribution of *normal* data.

    Set it at the Nth percentile, and roughly (100 - N)% of normal windows will
    trigger a false alarm — that is the trade-off, made explicit.
    """
    percentile = percentile if percentile is not None else config.THRESHOLD_PERCENTILE
    return float(np.percentile(normal_scores, percentile))


def main():
    if not os.path.exists(config.MODEL_PATH):
        raise SystemExit("No trained model found. Run `python train.py` first.")

    torch.manual_seed(config.SEED)
    splits, scaler, (series, labels) = data.prepare_benchmark()
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]

    print("Loading models…")
    autoencoder = model_module.load_model(config.MODEL_PATH)
    forest = baseline.fit_baseline(x_train, y_train)

    # --- scores -----------------------------------------------------------
    ae_val = autoencoder.reconstruction_error(torch.from_numpy(x_val)).numpy()
    ae_test = autoencoder.reconstruction_error(torch.from_numpy(x_test)).numpy()

    forest_val = baseline.scores(forest, x_val)
    forest_test = baseline.scores(forest, x_test)

    # --- thresholds, calibrated on normal validation data only ------------
    ae_threshold = calibrate(ae_val[y_val == 0])
    forest_threshold = calibrate(forest_val[y_val == 0])

    autoencoder_results = metrics_from_scores(ae_test, y_test, ae_threshold)
    forest_results = metrics_from_scores(forest_test, y_test, forest_threshold)

    # --- report -----------------------------------------------------------
    print()
    print("=" * 62)
    print("  RESULTS ON THE HELD-OUT TEST SPLIT")
    print("=" * 62)
    header = f"  {'model':<26}{'precision':>10}{'recall':>9}{'F1':>8}"
    print(header)
    print("  " + "-" * 58)
    for name, result in [("LSTM autoencoder (deep)", autoencoder_results),
                         ("Isolation Forest (classic)", forest_results)]:
        print(f"  {name:<26}{result['precision']:>10.3f}"
              f"{result['recall']:>9.3f}{result['f1']:>8.3f}")
    print("  " + "-" * 58)
    print(f"  test windows: {autoencoder_results['total_windows']}"
          f"   true anomalies: {autoencoder_results['true_anomalies']}")

    best = max(autoencoder_results["f1"], forest_results["f1"])
    winner = ("LSTM autoencoder (deep)" if autoencoder_results["f1"] >= forest_results["f1"]
              else "Isolation Forest (classic)")
    print(f"\n  Best F1: {best:.3f} — {winner}")
    print("\n  Precision = of the windows flagged, how many were real anomalies.")
    print("  Recall    = of the real anomalies, how many were caught.")

    report = {
        "autoencoder": autoencoder_results,
        "isolation_forest": forest_results,
        "winner": winner,
    }
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {config.REPORT_PATH}")


if __name__ == "__main__":
    main()
