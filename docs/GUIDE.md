# 📖 Project Guide — NeuroSense

Everything about this project, explained from the ground up. This is the most
advanced project in the portfolio, so the guide spends more time on the ideas.

**Repository:** `neurosense`
**What it is in one line:** a neural network that learns what "normal" looks
like for a sensor, then flags anything that does not fit — measured honestly
against a simpler classical method.

---

## 1. What problem does it solve?

A sensor produces a reading every few seconds, forever. Buried in that stream
are moments that matter: a spike, a slow drift, a sensor that has stopped
working. Staring at a chart will not find them — there is too much data and the
interesting parts are rare.

So you teach a model what normal looks like, and let it point at the rest.

The second, harder problem this project solves is: **does the fancy approach
actually work?** Most anomaly-detection projects stop at "I used Isolation
Forest". This one asks the follow-up question and answers it with real numbers,
including the cases where the fancy approach is *not* clearly better.

---

## 2. The big picture

```
   raw readings (temperature, humidity, light)
              │
              ▼
   cut into windows of 12 consecutive readings
              │
              ▼
   standardise (subtract mean, divide by std)
              │
              ▼
   ┌──────────────────────────────┐
   │      LSTM autoencoder        │
   │   encode  ──► bottleneck(6)  │
   │              decode          │
   │           reconstruction     │
   └──────────────┬───────────────┘
                  │
                  ▼
       reconstruction error
                  │
                  ▼
        above threshold? ──► ANOMALY
```

---

## 3. The central idea: learning by reconstruction

This is the one concept to understand, and it is genuinely elegant.

**Training.** Show the network thousands of *normal* windows. Ask it to
reproduce each one. It gets good at this — for normal data.

**The bottleneck.** The window (12 readings × 3 sensors = 36 numbers) is
squeezed down to just **6 numbers** before being rebuilt. The network physically
cannot copy the input through — there is not enough room. So it must learn the
*patterns* that matter, and throw away the noise.

**Detection.** Now show it something unusual — a spike, a drift. It has never
learned to represent that, so it reconstructs it badly. **The size of the
mistake is the anomaly score.**

Why this works so well: you never need labelled anomalies to train. You only
need normal data, which you have plenty of. That is what makes it practical.

---

## 4. Every technology used, and what it is

| Technology | What it actually is | Where it appears |
|---|---|---|
| **Python** | The language. | all files |
| **PyTorch** | The most widely used deep-learning library. It gives you the building blocks (layers, losses, optimisers) and — crucially — **automatic differentiation**: it works out the gradients for you. | `model.py`, `train.py` |
| **Neural network** | A stack of simple maths operations with adjustable numbers ("weights"). Training nudges the weights until the output is right. | `model.py` |
| **LSTM** | *Long Short-Term Memory* — a layer designed for sequences. Unlike a plain layer, it carries a memory of what it has seen, which is why it suits time-series data. | `model.py` |
| **Autoencoder** | A network trained to output its own input. Sounds pointless; is not, because of the bottleneck. | `model.py` |
| **Reconstruction error** | The difference between the input and what the network rebuilt. Mean squared error here. | `model.py` |
| **Epoch** | One full pass over the training data. | `train.py` |
| **Batch** | A small group of examples processed together. | `train.py` |
| **Loss** | A single number measuring how wrong the model is. Training tries to make it smaller. | `train.py` |
| **Backpropagation** | The algorithm that computes how much each weight contributed to the error. PyTorch does this when you call `.backward()`. | `train.py` |
| **Optimiser (Adam)** | The rule for adjusting weights, based on the gradients. | `train.py` |
| **Early stopping** | Stop training when validation stops improving, and keep the best version — not the last one. | `train.py` |
| **Validation split** | Data held back from training, used to judge the model and to set the alert threshold. | `data.py` |
| **Standardisation** | Rescaling each feature to mean 0 and standard deviation 1, so no sensor dominates just because its numbers are bigger. | `data.py` |
| **Isolation Forest** | The classical comparison. Finds odd rows by how quickly random splits isolate them. | `baseline.py` |
| **Precision / Recall / F1** | The metrics that matter here. Precision = of the flags raised, how many were real. Recall = of the real anomalies, how many were caught. F1 balances them. | `evaluate.py` |
| **NumPy** | Python's array library. Every number here is a NumPy array. | everywhere |
| **Matplotlib** | Draws the training curve. | `docs/training-curve.png` |
| **pytest** | The testing tool. | `tests/` |
| **GitHub Actions** | Runs the tests on every push, installing the CPU build of PyTorch. | `.github/workflows/` |

---

## 5. File-by-file: what each one does

### `config.py`
Every tunable number in one place: window size, hidden size, bottleneck size,
learning rate, epochs, patience, threshold percentile. **Read this first** — it
makes the rest of the code readable, because you already know what the numbers
mean.

### `data.py`
Builds the benchmark and prepares it. Three parts:

- `make_synthetic()` — creates a labelled dataset. Real sensor data has no
  ground truth, so to measure the model honestly we build a series where we
  *know* which readings are anomalous, because we put them there. Three
  realistic types: **spikes**, **level shifts**, and **stuck sensors**.
- `windowise()` — cuts the series into overlapping windows. A window is labelled
  anomalous if *any* reading inside it is.
- `prepare_benchmark()` — splits by **time**, not at random. Training on the
  future to detect the past would flatter the model. The scaler is fitted on the
  training portion only, so no information leaks.

### `model.py`
The LSTM autoencoder, about 60 lines. `encode()` compresses a window to a
6-number vector; `forward()` expands it back. `reconstruction_error()` returns
the per-window error — the anomaly score.

### `train.py`
The training loop. The pattern to notice is only four lines:

```python
optimiser.zero_grad()        # forget the last gradients
loss = loss_fn(model(x), x)  # how wrong are we?
loss.backward()              # work out who to blame
optimiser.step()             # nudge the weights
```

Everything else in the file is bookkeeping: measuring validation loss, keeping
the best weights, stopping early.

Note that **only normal windows go into training**. Anomalies are deliberately
excluded, so the model never learns to reconstruct them.

### `baseline.py`
The Isolation Forest comparison — the same model the ESP32 project uses. It is
here for one reason: a number on its own means nothing. "F1 of 0.79" is only
interesting next to what a simpler method achieves on identical data with an
identical threshold rule.

### `evaluate.py`
The honest part. Both models are scored identically: fit on normal training
windows, calibrate the threshold on the **validation** split, and measure on the
**test** split.

Tuning the threshold on the test split is the most common way people
accidentally cheat and report numbers that do not hold up. This file does not.

### `detect.py`
The practical tool. Points at a real sensor database and reports which windows
look anomalous, with timestamps. With no labels to learn from, the honest
question is "show me the most unusual 1%", so the threshold is a percentile of
the scores.

### `tests/test_neurosense.py`
15 tests. The most interesting are the data ones: an anomaly detector is only
as trustworthy as its benchmark, so the synthetic data, its anomaly rate, its
reproducibility, and the time-ordered split are all checked explicitly. Also
covered: the model's shapes, a save/load round-trip, the metrics maths against a
hand-computed case, and that training actually reduces loss.

### The other files
- `README.md` — features, the results table, quick start
- `docs/training-curve.png` — the loss curves from a real run
- `LICENSE` — MIT

---

## 6. The result, and how to read it

| Model | Precision | Recall | F1 |
|---|---|---|---|
| LSTM autoencoder (deep) | 0.649 | **1.000** | **0.787** |
| Isolation Forest (classical) | **0.862** | 0.676 | 0.758 |

*1,489 test windows, 37 genuinely anomalous.*

- The autoencoder **caught every anomaly** but raised more false alarms.
- The forest was **more precise** but **missed a third** of the real anomalies.

Neither is simply better. If a miss is expensive (equipment overheating) you
want the autoencoder. If a false alarm is expensive (paging someone at 3am) you
want the forest. **Being able to say that is worth more than claiming a
winner.**

---

## 7. How to run it

```bash
pip install -r requirements.txt

python train.py                          # ~20 seconds on a CPU
python evaluate.py                       # the head-to-head comparison
python detect.py --db artifacts/test_sensors.db
```

To score real data from the ESP32 project:

```bash
SENSOR_DB=../esp32-environment-monitor/data/sensors.db python detect.py
```

---

## 8. Concepts worth understanding (in this order)

1. **A NumPy array** — a grid of numbers
2. **`_normal_series()` in `data.py`** — a sine wave plus noise. That is all
   "normal" is
3. **`windowise()`** — ten lines, and the heart of the data pipeline
4. **The four-line training loop** — forward, loss, backward, step
5. **Precision vs recall** — why both matter and why they trade off

Items 1–3 are approachable early. Item 4 is the one to stare at until it clicks —
once it does, most of deep learning stops looking like magic.

---

## 9. Common problems and what they mean

| Symptom | What is happening | Fix |
|---|---|---|
| `No trained model found` | Training has not been run | `python train.py` |
| `ModuleNotFoundError: torch` | PyTorch not installed | `pip install -r requirements.txt` (or the CPU wheel, which is much smaller) |
| Training is slow | PyTorch installed the CUDA build | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Loss does not fall | Learning rate too high, or data not scaled | Check `config.py` — the scaler is applied in `data.py` |
| Everything is flagged as anomalous | Threshold too low | Raise `THRESHOLD_PERCENTILE` in `config.py` |

---

## 10. What you would say about this project in an interview

> "It's an LSTM autoencoder that learns to reconstruct normal sensor windows;
> the reconstruction error is the anomaly score. The part I'd emphasise is the
> evaluation — I built a labelled benchmark with three realistic anomaly types,
> split it by time, calibrated the threshold on validation, and compared it
> against an Isolation Forest on the same test set. The autoencoder caught every
> anomaly but had lower precision, so it's a recall-versus-precision trade-off
> rather than a clean win."

Mentioning the **trade-off** rather than a victory is what marks this out as
real work rather than a tutorial you followed.

---

## 11. Where this project goes next

See the roadmap in the README: a Transformer encoder to compare against the
LSTM, per-feature errors to say *which* sensor went wrong, and quantising the
model to run on the ESP32 itself.
