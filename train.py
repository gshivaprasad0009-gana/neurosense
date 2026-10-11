"""
Training.

Trains the autoencoder on normal windows and saves the best model, the scaler
and the loss history. Three things here are worth pointing out, because they
are the difference between a model that works and one that only looks like it:

* **A validation split.** The model is judged on data it never trained on, so
  the loss curve tells you about generalisation, not memorisation.
* **Early stopping.** Training stops when validation loss stops improving, and
  the *best* weights are kept — not the last ones.
* **Time-ordered splits.** Windows are shuffled for batching (fine), but the
  split itself is chronological, so the model is never trained on the future.

Usage:
    python train.py
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
import data
import model as model_module


def set_seed(seed: int) -> None:
    """Make runs reproducible — same seed, same result."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(xs, ys, shuffle: bool, batch_size: int) -> DataLoader:
    # Only normal windows go into training: the model must learn "normal",
    # so anomalies are deliberately excluded from the training set.
    normal = xs[ys == 0]
    dataset = TensorDataset(torch.from_numpy(normal))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model, xs, ys, loss_fn) -> float:
    """Mean reconstruction loss over the *normal* windows of a split."""
    normal = torch.from_numpy(xs[ys == 0])
    if len(normal) == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(normal), normal).item())


def main():
    set_seed(config.SEED)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    print("Preparing data…")
    splits, scaler, (series, labels) = data.prepare_benchmark()
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]

    print(f"  training windows : {len(x_train):>5}  ({int(y_train.sum())} anomalous, excluded)")
    print(f"  validation       : {len(x_val):>5}  ({int(y_val.sum())} anomalous)")
    print(f"  test             : {len(splits['test'][0]):>5}  ({int(splits['test'][1].sum())} anomalous)")

    train_loader = make_loader(x_train, y_train, shuffle=True, batch_size=config.BATCH_SIZE)

    model = model_module.build_model(n_features=len(config.FEATURES))
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val = float("inf")
    best_state = None
    history = {"train": [], "val": []}
    epochs_without_improvement = 0
    started = time.time()

    print(f"\nTraining for up to {config.EPOCHS} epochs (patience {config.PATIENCE})…")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running = 0.0
        for (batch,) in train_loader:
            optimiser.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(batch)
        train_loss = running / len(train_loader.dataset)

        val_loss = evaluate_loss(model, x_val, y_val, loss_fn)
        history["train"].append(round(train_loss, 6))
        history["val"].append(round(val_loss, 6))

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        flag = " *" if improved else ""
        print(f"  epoch {epoch:>3}  train {train_loss:.5f}  val {val_loss:.5f}{flag}")

        if epochs_without_improvement >= config.PATIENCE:
            print(f"  no improvement for {config.PATIENCE} epochs — stopping early.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)      # keep the best, not the last

    model_module.save_model(model, config.MODEL_PATH)
    scaler.save(config.SCALER_PATH)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - started
    print(f"\nDone in {elapsed:.1f}s. Best validation loss: {best_val:.6f}")
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
