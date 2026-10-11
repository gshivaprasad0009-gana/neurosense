"""
Shared settings for NeuroSense.

Everything tunable lives here so the scripts stay readable and you can
experiment from one place.
"""

import os

# --- data -----------------------------------------------------------------
FEATURES = ["temperature", "humidity", "light"]
WINDOW = 12              # readings per example — one hour at five-minute intervals
ANOMALY_RATE = 0.02      # fraction of the synthetic benchmark that is anomalous
N_NORMAL = 6000          # length of the synthetic normal series
SEED = 42

# --- model ----------------------------------------------------------------
HIDDEN = 16              # size of the LSTM hidden state
LATENT = 6               # size of the bottleneck the model must squeeze through
DROPOUT = 0.1

# --- training -------------------------------------------------------------
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 8             # early stopping: epochs without improvement
VAL_FRACTION = 0.2

# --- detection ------------------------------------------------------------
# A reading is flagged when its reconstruction error exceeds this percentile
# of the *validation* errors. Calibrating on validation data (not test data)
# is what keeps the evaluation honest.
THRESHOLD_PERCENTILE = 99.0

# --- paths ----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "artifacts")
MODEL_PATH = os.path.join(DATA_DIR, "autoencoder.pt")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
REPORT_PATH = os.path.join(DATA_DIR, "evaluation.json")
CURVE_PATH = os.path.join(HERE, "docs", "training-curve.png")

# The ESP32 project writes to this schema, so you can point NeuroSense at its
# real data with SENSOR_DB=../esp32-environment-monitor/data/sensors.db
SENSOR_DB = os.environ.get("SENSOR_DB", "")
