"""
The classical baseline — Isolation Forest.

This is the model the ESP32 Environment Monitor already uses. It is here for
one reason: **a number on its own means nothing.** "The autoencoder got an
F1 of 0.83" is only interesting next to what a simpler method achieves on the
exact same data, with the exact same thresholding rule.

Comparing against a baseline is what separates an evaluation from a demo.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

import config


def flatten(xs: np.ndarray) -> np.ndarray:
    """Isolation Forest wants flat rows, so each window becomes one long vector."""
    return xs.reshape(len(xs), -1)


def fit_baseline(xs: np.ndarray, ys: np.ndarray, contamination: float = 0.02):
    """Fit on the normal training windows only, exactly like the autoencoder."""
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=config.SEED
    )
    model.fit(flatten(xs[ys == 0]))
    return model


def scores(model, xs: np.ndarray) -> np.ndarray:
    """Anomaly scores where **higher means more anomalous** (so it matches the
    autoencoder's reconstruction error)."""
    return -model.score_samples(flatten(xs))
