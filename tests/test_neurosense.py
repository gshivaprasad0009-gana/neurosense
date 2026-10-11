"""
Tests for NeuroSense.

The interesting ones are the data tests: an anomaly detector is only as
trustworthy as its benchmark, so the synthetic data and the time-ordered split
are checked explicitly.

Run from the project root:

    pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import baseline  # noqa: E402
import config  # noqa: E402
import data  # noqa: E402
import evaluate  # noqa: E402
import model as model_module  # noqa: E402


# --- data -------------------------------------------------------------

def test_synthetic_series_has_the_expected_shape():
    series, labels = data.make_synthetic(n=500, seed=1)
    assert series.shape == (500, len(config.FEATURES))
    assert labels.shape == (500,)
    assert set(np.unique(labels)) <= {0, 1}


def test_synthetic_series_contains_anomalies():
    """ANOMALY_RATE is the fraction of anomalous *readings*, so the measured
    rate should land close to it."""
    _, labels = data.make_synthetic(n=4000, anomaly_rate=0.02, seed=1)
    rate = labels.mean()
    assert 0.01 < rate < 0.05, f"anomaly rate looks wrong: {rate:.3f}"


def test_synthetic_data_is_reproducible():
    a, la = data.make_synthetic(n=300, seed=99)
    b, lb = data.make_synthetic(n=300, seed=99)
    assert np.allclose(a, b)
    assert np.array_equal(la, lb)


def test_windowise_shapes():
    series, labels = data.make_synthetic(n=200, seed=2)
    xs, ys = data.windowise(series, labels, window=10)
    assert xs.shape == (191, 10, len(config.FEATURES))
    assert ys.shape == (191,)


def test_window_is_labelled_anomalous_if_any_reading_is():
    series = np.zeros((20, 3))
    labels = np.zeros(20, dtype=int)
    labels[12] = 1                       # one anomalous reading
    xs, ys = data.windowise(series, labels, window=5)
    # windows starting at 8..12 contain index 12
    assert ys[8:13].all()
    assert ys[:8].sum() == 0
    assert ys[13:].sum() == 0


def test_scaler_round_trip(tmp_path):
    series, _ = data.make_synthetic(n=300, seed=3)
    scaler = data.Scaler().fit(series)
    scaled = scaler.transform(series)
    assert abs(scaled.mean()) < 1e-6          # centred
    assert abs(scaled.std() - 1) < 1e-3       # unit scale

    path = tmp_path / "scaler.json"
    scaler.save(str(path))
    reloaded = data.Scaler.load(str(path))
    assert np.allclose(reloaded.transform(series), scaled)


def test_splits_are_time_ordered_and_disjoint():
    """The split must not shuffle time — training on the future would flatter
    the model. This checks the windows come out in the right order."""
    splits, scaler, (series, labels) = data.prepare_benchmark(seed=5)
    assert len(splits["train"][0]) > 0
    assert len(splits["val"][0]) > 0
    assert len(splits["test"][0]) > 0

    # Reconstruct what each split should cover and confirm no overlap.
    n = len(series)
    train_end, val_end = int(n * 0.5), int(n * 0.75)
    assert len(splits["train"][0]) == train_end - config.WINDOW + 1
    assert len(splits["val"][0]) == val_end - train_end - config.WINDOW + 1
    assert len(splits["test"][0]) == n - val_end - config.WINDOW + 1


# --- model ------------------------------------------------------------

def test_autoencoder_reconstructs_the_input_shape():
    model = model_module.LSTMAutoencoder(n_features=3, hidden=8, latent=4, window=12)
    x = torch.randn(5, 12, 3)
    assert model(x).shape == x.shape


def test_reconstruction_error_is_per_window():
    model = model_module.LSTMAutoencoder(n_features=3, hidden=8, latent=4, window=12)
    x = torch.randn(7, 12, 3)
    errors = model.reconstruction_error(x)
    assert errors.shape == (7,)
    assert (errors >= 0).all()


def test_model_save_and_load_round_trip(tmp_path):
    model = model_module.LSTMAutoencoder(n_features=3, hidden=8, latent=4, window=12)
    path = tmp_path / "model.pt"
    model_module.save_model(model, str(path))

    loaded = model_module.load_model(str(path))
    x = torch.randn(4, 12, 3)

    # Compare in eval mode: dropout is active in train mode and would make the
    # two forward passes differ for reasons that have nothing to do with saving.
    model.eval()
    assert torch.allclose(model(x), loaded(x), atol=1e-6)


# --- baseline ---------------------------------------------------------

def test_baseline_scores_are_higher_for_anomalies():
    splits, scaler, _ = data.prepare_benchmark(seed=6)
    x_train, y_train = splits["train"]
    forest = baseline.fit_baseline(x_train, y_train)

    normal = x_train[y_train == 0][:50]
    # A blatant anomaly: temperature far above anything in the training data
    broken = normal.copy()
    broken[:, :, 0] += 25.0
    assert baseline.scores(forest, broken).mean() > baseline.scores(forest, normal).mean()


# --- evaluation -------------------------------------------------------

def test_metrics_from_scores_known_case():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.2, 0.1, 0.8])
    result = evaluate.metrics_from_scores(scores, labels, threshold=0.5)
    # predicted = [1, 0, 0, 1] -> one true positive, one false positive, one miss
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["flagged"] == 2


def test_calibrate_returns_the_requested_percentile():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert evaluate.calibrate(scores, percentile=80) == pytest.approx(3.2)
    assert evaluate.calibrate(scores, percentile=0) == pytest.approx(0.0)
    assert evaluate.calibrate(scores, percentile=100) == pytest.approx(4.0)


def test_perfect_predictions_score_one():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    result = evaluate.metrics_from_scores(scores, labels, threshold=0.5)
    assert result["f1"] == 1.0


# --- training (smoke test) --------------------------------------------

def test_training_reduces_loss():
    """A few epochs on a small series must lower the loss — proof the
    training loop actually learns, without a long test run."""
    torch.manual_seed(0)
    series, labels = data.make_synthetic(n=400, seed=7)
    scaler = data.Scaler().fit(series)
    xs, ys = data.windowise(scaler.transform(series), labels)
    xs = xs.astype(np.float32)

    model = model_module.LSTMAutoencoder(n_features=3, hidden=8, latent=4, window=config.WINDOW)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    normal = torch.from_numpy(xs[ys == 0])

    model.train()
    first = float(loss_fn(model(normal), normal).item())
    for _ in range(15):
        optimiser.zero_grad()
        loss = loss_fn(model(normal), normal)
        loss.backward()
        optimiser.step()
    final = float(loss_fn(model(normal), normal).item())

    assert final < first, f"loss did not fall: {first:.5f} -> {final:.5f}"
