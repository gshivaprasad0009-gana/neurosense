"""
The model — an LSTM autoencoder.

The idea behind anomaly detection by reconstruction:

1. Train the network to reproduce **normal** windows as accurately as it can.
2. Because it only ever saw normal behaviour, it gets good at normal windows
   and stays clumsy on anything unusual.
3. So the *reconstruction error* is the anomaly score. High error means
   "this does not look like anything I was trained on".

The bottleneck (`LATENT`) is what makes this work: the network must squeeze
each window through a few numbers and rebuild it, so it can only preserve the
patterns it learned. It cannot simply copy the input through.
"""

import torch
import torch.nn as nn

import config


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = None, latent: int = None,
                 window: int = None, dropout: float = None):
        super().__init__()
        hidden = hidden or config.HIDDEN
        latent = latent or config.LATENT
        self.window = window or config.WINDOW
        self.n_features = n_features
        self.latent = latent

        # Encoder: read the window, keep a summary of it
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)

        # Decoder: expand the summary back into a full window
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.to_output = nn.Linear(hidden, n_features)

        self.dropout = nn.Dropout(dropout if dropout is not None else config.DROPOUT)
        self.activation = nn.ReLU()

    def encode(self, x):
        """Compress a window (batch, window, features) to (batch, latent)."""
        _, (hidden_state, _) = self.encoder(x)
        return self.to_latent(self.activation(hidden_state[-1]))

    def forward(self, x):
        batch = x.size(0)
        latent = self.encode(x)

        # Repeat the latent vector across the window length, then decode
        expanded = self.activation(self.from_latent(latent)).unsqueeze(1)
        expanded = expanded.repeat(1, self.window, 1)
        decoded, _ = self.decoder(expanded)
        return self.to_output(self.dropout(decoded))

    def reconstruction_error(self, x):
        """Per-window mean squared error — the anomaly score.

        Returned per window (not per reading) so it lines up with the labels.
        """
        with torch.no_grad():
            rebuilt = self.forward(x)
            return ((rebuilt - x) ** 2).mean(dim=(1, 2))


def build_model(n_features: int) -> LSTMAutoencoder:
    return LSTMAutoencoder(n_features)


def save_model(model: LSTMAutoencoder, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": model.n_features,
        "hidden": model.encoder.hidden_size,
        "latent": model.latent,
        "window": model.window,
    }, path)


def load_model(path: str) -> LSTMAutoencoder:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = LSTMAutoencoder(
        checkpoint["n_features"],
        hidden=checkpoint["hidden"],
        latent=checkpoint["latent"],
        window=checkpoint["window"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
