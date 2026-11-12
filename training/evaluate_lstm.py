
"""
Evaluate the existing BiLSTM checkpoint without retraining.
Uses the same Indian held-out test dataset as ResNet18.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from tqdm import tqdm

# --------------------------------------------------
# PATHS AND IMPORTS
# --------------------------------------------------

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent

sys.path.insert(0, str(TRAINING_DIR))

from lstm_model import BiLSTMClassifier
from dataset_loader import test_loader, test_dataset
from train_config import DEVICE

CHECKPOINT_PATH = TRAINING_DIR / "checkpoints" / "lstm_best.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

METRICS_PATH = RESULTS_DIR / "lstm_test_metrics.txt"
CONFUSION_PATH = RESULTS_DIR / "lstm_confusion_matrix.csv"
ARRAY_PATH = RESULTS_DIR / "lstm_test_arrays.npz"

# --------------------------------------------------
# MODEL CONFIGURATION
# Must match the saved checkpoint architecture.
# --------------------------------------------------

INPUT_SIZE = 64
TIME_STEPS = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)

print("=" * 60)
print("BiLSTM CHECKPOINT EVALUATION")
print("=" * 60)

print("Device:", DEVICE)
print("Test samples:", len(test_dataset))

# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

assert checkpoint["lstm_input_size"] == INPUT_SIZE
assert checkpoint["lstm_time_steps"] == TIME_STEPS
assert checkpoint["lstm_hidden"] == HIDDEN_SIZE
assert checkpoint["lstm_layers"] == NUM_LAYERS

model = BiLSTMClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=0.3,
    fc_dropout=0.4,
    num_classes=2,
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint loaded successfully.")
print("Best epoch:", checkpoint["epoch"])
print("Best validation loss:", checkpoint["best_val_loss"])

# --------------------------------------------------
# MEL IMAGE TO LSTM SEQUENCE
# --------------------------------------------------

def images_to_lstm_input(images):

    resized = F.interpolate(
        images,
        size=(INPUT_SIZE, TIME_STEPS),
        mode="bilinear",
        align_corners=False,
    )

    squeezed = resized.squeeze(1)

    return squeezed.permute(0, 2, 1)

# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

all_labels = []
all_predictions = []
all_fake_probabilities = []

# Warm-up for more consistent timing
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(DEVICE)
        sequences = images_to_lstm_input(images)
        model(sequences)
        break

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

start_time = time.perf_counter()

with torch.no_grad():

    for images, labels in tqdm(test_loader, desc="Evaluating BiLSTM"):

        images = images.to(DEVICE)

        sequences = images_to_lstm_input(images)

        logits = model(sequences)

        probabilities = torch.softmax(logits, dim=1)

        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_fake_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

end_time = time.perf_counter()

# --------------------------------------------------
# ARRAYS AND METRICS
# --------------------------------------------------

labels = np.array(all_labels)
predictions = np.array(all_predictions)
fake_probabilities = np.array(all_fake_probabilities)

assert len(labels) == len(test_dataset)
assert len(predictions) == len(test_dataset)

accuracy = accuracy_score(labels, predictions)

precision = precision_score(
    labels, predictions, pos_label=1, zero_division=0
)

recall = recall_score(
    labels, predictions, pos_label=1, zero_division=0
)

f1 = f1_score(
    labels, predictions, pos_label=1, zero_division=0
)

try:
    auc = roc_auc_score(labels, fake_probabilities)
except ValueError:
    auc = float("nan")

cm = confusion_matrix(labels, predictions, labels=[0, 1])

total_time = end_time - start_time
per_sample_ms = total_time / len(labels) * 1000

model_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 ** 2)

report = classification_report(
    labels,
    predictions,
    labels=[0, 1],
    target_names=["REAL", "FAKE"],
    zero_division=0,
)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

summary = f"""
BI-LSTM TEST EVALUATION
=======================

Checkpoint epoch: {checkpoint['epoch']}
Best validation loss: {checkpoint['best_val_loss']:.6f}

Test samples: {len(labels)}
Device: {DEVICE}

Accuracy: {accuracy:.4f}
Precision (FAKE): {precision:.4f}
Recall (FAKE): {recall:.4f}
F1-score (FAKE): {f1:.4f}
ROC-AUC: {auc:.4f}

Total inference time: {total_time:.4f} seconds
Per-sample inference time: {per_sample_ms:.4f} ms
Checkpoint size: {model_size_mb:.2f} MB

Confusion matrix:
{cm}

Classification report:
{report}
"""

print(summary)

METRICS_PATH.write_text(summary, encoding="utf-8")

pd.DataFrame(
    cm,
    index=["Actual_REAL", "Actual_FAKE"],
    columns=["Predicted_REAL", "Predicted_FAKE"],
).to_csv(CONFUSION_PATH)

np.savez(
    ARRAY_PATH,
    labels=labels,
    predictions=predictions,
    fake_probabilities=fake_probabilities,
    inference_time_total_s=np.float64(total_time),
    inference_time_per_sample_ms=np.float64(per_sample_ms),
    model_size_mb=np.float64(model_size_mb),
)

print("Saved:", METRICS_PATH)
print("Saved:", CONFUSION_PATH)
print("Saved:", ARRAY_PATH)

print("\nBiLSTM evaluation completed successfully.")
