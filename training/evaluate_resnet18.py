"""
evaluate_resnet18.py
--------------------
Loads the best ResNet18 checkpoint (training/checkpoints/resnet18_best.pth)
and evaluates it on the HELD-OUT TEST SET only.

This script does NOT train or retrain anything.
It does NOT touch the checkpoint or any existing files.
All outputs are written to the results/ folder.

Label mapping  : real=0, fake=1
Positive class : FAKE (label 1)
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

# ---------------------------------------------------------------------------
# Make sure imports from this package work when run from project root or
# from the training/ directory.
# ---------------------------------------------------------------------------
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from resnet18_model import create_resnet18
from dataset_loader import test_loader, test_dataset
from train_config import DEVICE


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = TRAINING_DIR / "checkpoints" / "resnet18_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_OUT = RESULTS_DIR / "resnet18_test_metrics.txt"
CONFUSION_OUT = RESULTS_DIR / "resnet18_confusion_matrix.csv"
NUMPY_OUT = RESULTS_DIR / "resnet18_test_arrays.npz"   # labels + probs for compare script

# Input size must match what the model was trained with (stored in checkpoint).
INPUT_SIZE = (224, 336)


# ---------------------------------------------------------------------------
# GUARD: refuse to overwrite existing results silently
# ---------------------------------------------------------------------------

for path in (METRICS_OUT, CONFUSION_OUT, NUMPY_OUT):
    if path.exists():
        print(f"WARNING: {path.name} already exists in results/. It will be overwritten.")


# ---------------------------------------------------------------------------
# DEVICE CHECK
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  RESNET18 - TEST SET EVALUATION")
print("=" * 60)
print(f"Device          : {DEVICE}")
print(f"Checkpoint      : {CHECKPOINT_PATH}")
print(f"Test samples    : {len(test_dataset)}")

if not torch.cuda.is_available() and str(DEVICE) == "cuda":
    print("\nERROR: CUDA was requested but is not available on this machine.")
    print("Set DEVICE to 'cpu' in train_config.py or install a CUDA-enabled PyTorch.")
    sys.exit(1)

print(f"Running on      : {DEVICE}")


# ---------------------------------------------------------------------------
# LOAD CHECKPOINT
# ---------------------------------------------------------------------------

print("\nLoading checkpoint ...")

if not CHECKPOINT_PATH.exists():
    print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    sys.exit(1)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)

print(f"  Saved epoch     : {checkpoint['epoch']}")
print(f"  Best val loss   : {checkpoint['best_val_loss']:.6f}")
print(f"  Input size      : {checkpoint['input_size']}")
print(f"  class_to_label  : {checkpoint['class_to_label']}")

# Confirm the input size matches what we will feed the model.
assert checkpoint["input_size"] == INPUT_SIZE, (
    f"Checkpoint input_size {checkpoint['input_size']} != expected {INPUT_SIZE}"
)

model = create_resnet18(pretrained=False).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Quick sanity forward pass before we touch real data.
with torch.no_grad():
    dummy = torch.randn(2, 1, *INPUT_SIZE, device=DEVICE)
    out = model(dummy)
assert out.shape == torch.Size([2, 2]), f"Unexpected model output shape: {out.shape}"
print("  Sanity check    : OK - model output shape [2, 2]")


# ---------------------------------------------------------------------------
# TEST EVALUATION
# ---------------------------------------------------------------------------

print("\nRunning evaluation on test set ...")

all_labels: list = []
all_predictions: list = []
all_fake_probabilities: list = []

# --- Inference timing (wall-clock over the entire test set) ---
torch.manual_seed(42)   # reproducible dropout behaviour (eval mode = no dropout, but good habit)
timing_start = time.perf_counter()

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing ResNet18"):

        images = images.to(DEVICE)

        # Resize to the training input size (same as train_resnet18.py).
        images = F.interpolate(
            images,
            size=INPUT_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())               # labels stay on CPU
        all_predictions.extend(predictions.cpu().tolist())
        all_fake_probabilities.extend(probabilities[:, 1].cpu().tolist())

timing_end = time.perf_counter()
total_inference_time = timing_end - timing_start
per_sample_inference_ms = (total_inference_time / len(all_labels)) * 1000.0

print(f"\nTotal inference time : {total_inference_time:.2f} s")
print(f"Per-sample time      : {per_sample_inference_ms:.3f} ms")

# Convert to arrays for metrics.
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_fake_probabilities = np.array(all_fake_probabilities)

# Guard: make sure we actually ran on the right number of samples.
assert len(all_labels) == len(test_dataset), (
    f"Label count mismatch: {len(all_labels)} vs {len(test_dataset)}"
)


# ---------------------------------------------------------------------------
# COMPUTE METRICS  (positive class = FAKE = 1)
# ---------------------------------------------------------------------------

accuracy  = accuracy_score(all_labels, all_predictions)

precision = precision_score(all_labels, all_predictions, pos_label=1, zero_division=0)
recall    = recall_score(   all_labels, all_predictions, pos_label=1, zero_division=0)
f1        = f1_score(       all_labels, all_predictions, pos_label=1, zero_division=0)

precision_macro  = precision_score(all_labels, all_predictions, average="macro",    zero_division=0)
recall_macro     = recall_score(   all_labels, all_predictions, average="macro",    zero_division=0)
f1_macro         = f1_score(       all_labels, all_predictions, average="macro",    zero_division=0)

precision_weighted = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
recall_weighted    = recall_score(   all_labels, all_predictions, average="weighted", zero_division=0)
f1_weighted        = f1_score(       all_labels, all_predictions, average="weighted", zero_division=0)

try:
    auc = roc_auc_score(all_labels, all_fake_probabilities)
except ValueError:
    auc = float("nan")
    print("WARNING: ROC-AUC could not be computed (only one class present in labels?).")

cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=["REAL", "FAKE"],
    zero_division=0,
)

# Model size on disk.
model_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 ** 2)


# ---------------------------------------------------------------------------
# PRINT AND SAVE RESULTS
# ---------------------------------------------------------------------------

summary = (
    "=" * 60 + "\n"
    "  RESNET18 - TEST SET METRICS\n"
    "=" * 60 + "\n\n"
    f"Checkpoint          : {CHECKPOINT_PATH.name}\n"
    f"Checkpoint epoch    : {checkpoint['epoch']}\n"
    f"Best val loss       : {checkpoint['best_val_loss']:.6f}\n"
    f"Test samples        : {len(all_labels)}\n"
    f"Device              : {DEVICE}\n\n"
    "--- Per-class (positive = FAKE, label 1) ---\n"
    f"Accuracy            : {accuracy:.4f}\n"
    f"Precision (FAKE)    : {precision:.4f}\n"
    f"Recall    (FAKE)    : {recall:.4f}\n"
    f"F1-score  (FAKE)    : {f1:.4f}\n\n"
    "--- Macro average ---\n"
    f"Precision (macro)   : {precision_macro:.4f}\n"
    f"Recall    (macro)   : {recall_macro:.4f}\n"
    f"F1-score  (macro)   : {f1_macro:.4f}\n\n"
    "--- Weighted average ---\n"
    f"Precision (weighted): {precision_weighted:.4f}\n"
    f"Recall    (weighted): {recall_weighted:.4f}\n"
    f"F1-score  (weighted): {f1_weighted:.4f}\n\n"
    f"ROC-AUC             : {auc:.4f}\n\n"
    "--- Inference ---\n"
    f"Total inference time: {total_inference_time:.2f} s\n"
    f"Per-sample time     : {per_sample_inference_ms:.3f} ms\n"
    f"Checkpoint size     : {model_size_mb:.1f} MB\n\n"
    "--- Confusion matrix (rows=actual, cols=predicted) ---\n"
    f"                Pred REAL   Pred FAKE\n"
    f"  Actual REAL : {cm[0,0]:>9}   {cm[0,1]:>9}\n"
    f"  Actual FAKE : {cm[1,0]:>9}   {cm[1,1]:>9}\n\n"
    "--- Full classification report ---\n"
    + report
)

print("\n" + summary)

METRICS_OUT.write_text(summary, encoding="utf-8")
print(f"Metrics saved to   : {METRICS_OUT}")

pd.DataFrame(
    cm,
    index=["Actual_REAL", "Actual_FAKE"],
    columns=["Predicted_REAL", "Predicted_FAKE"],
).to_csv(CONFUSION_OUT)
print(f"Confusion matrix   : {CONFUSION_OUT}")

# Save raw arrays so compare_models.py can use them directly.
np.savez(
    NUMPY_OUT,
    labels=all_labels,
    predictions=all_predictions,
    fake_probabilities=all_fake_probabilities,
    inference_time_total_s=np.float64(total_inference_time),
    inference_time_per_sample_ms=np.float64(per_sample_inference_ms),
    model_size_mb=np.float64(model_size_mb),
)
print(f"Arrays saved to    : {NUMPY_OUT}")

print("\n" + "=" * 60)
print("  EVALUATION COMPLETE")
print("=" * 60)
