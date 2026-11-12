"""
train_lstm.py
-------------
Trains the BiLSTM deepfake voice classifier on the Indian Mel-spectrogram
dataset and evaluates it on the HELD-OUT TEST SET.

This script reuses:
  - training/dataset_loader.py  -> same split CSV, same loaders, same normalization
  - training/train_config.py    -> class weights, learning rate, device
  - training/lstm_model.py      -> BiLSTMClassifier architecture
Label mapping  : real=0, fake=1
Positive class : FAKE (label 1)

Saved outputs (all in training/ and results/ - nothing existing is overwritten):
  training/checkpoints/lstm_best.pth
  training/runs/lstm_history.csv
  results/lstm_test_metrics.txt
  results/lstm_confusion_matrix.csv
  results/lstm_test_arrays.npz    <- raw arrays for compare_models.py"""

from pathlib import Path
import sys
import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
# Resolve imports regardless of working directory
# ---------------------------------------------------------------------------
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from lstm_model import BiLSTMClassifier
from dataset_loader import (
    train_loader,
    validation_loader,
    test_loader,
    train_dataset,
    validation_dataset,
    test_dataset,
)
from train_config import (
    CLASS_WEIGHTS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    DEVICE,
)


# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# DEVICE CHECK
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  LSTM - TRAINING AND EVALUATION")
print("=" * 60)
print(f"Device              : {DEVICE}")
print(f"Train samples       : {len(train_dataset)}")
print(f"Validation samples  : {len(validation_dataset)}")
print(f"Test samples        : {len(test_dataset)}")

if not torch.cuda.is_available() and str(DEVICE) == "cuda":
    print("\nERROR: CUDA was requested in train_config.py but is not available.")
    print("Set DEVICE = torch.device('cpu') or install a CUDA-enabled PyTorch build.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
RUNS_DIR = TRAINING_DIR / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "lstm_best.pth"
HISTORY_PATH    = RUNS_DIR / "lstm_history.csv"
METRICS_PATH    = RESULTS_DIR / "lstm_test_metrics.txt"
CONFUSION_PATH  = RESULTS_DIR / "lstm_confusion_matrix.csv"
NUMPY_OUT       = RESULTS_DIR / "lstm_test_arrays.npz"


# ---------------------------------------------------------------------------
# LSTM HYPERPARAMETERS
# ---------------------------------------------------------------------------

# Input to the LSTM: image resized from [1, 400, 600] to [1, F, T] then
# permuted to [batch, T, F].  We use F=64 frequency bins and T=128 time steps.
#
# Orientation verification:
#   PIL loads image as (W=600, H=400).  ToTensor gives [C=1, H=400, W=600].
#   After F.interpolate to (H_new=64, W_new=128):
#     tensor shape = [C=1, freq=64, time=128]
#   Squeeze channel -> [freq=64, time=128]
#   Permute(1, 0)   -> [time=128, freq=64]
#   Batch           -> [batch, T=128, F=64]
#
LSTM_INPUT_SIZE  = 64    # F: frequency bins
LSTM_TIME_STEPS  = 128   # T: time steps
LSTM_HIDDEN      = 128   # hidden units per direction
LSTM_LAYERS      = 2     # stacked layers
LSTM_DROPOUT     = 0.3   # between LSTM layers
LSTM_FC_DROPOUT  = 0.4   # before final FC
EARLY_STOPPING_PATIENCE = 7


print(f"\nLSTM input shape    : [batch, T={LSTM_TIME_STEPS}, F={LSTM_INPUT_SIZE}]")
print(f"BiLSTM hidden       : {LSTM_HIDDEN}  (x2 directions = {2*LSTM_HIDDEN})")
print(f"LSTM layers         : {LSTM_LAYERS}")
print(f"Early stopping      : patience={EARLY_STOPPING_PATIENCE}")


# ---------------------------------------------------------------------------
# MODEL, CRITERION, OPTIMIZER, SCHEDULER
# ---------------------------------------------------------------------------

model = BiLSTMClassifier(
    input_size=LSTM_INPUT_SIZE,
    hidden_size=LSTM_HIDDEN,
    num_layers=LSTM_LAYERS,
    dropout=LSTM_DROPOUT,
    fc_dropout=LSTM_FC_DROPOUT,
    num_classes=2,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters    : {total_params:,}")

criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
)


# ---------------------------------------------------------------------------
# RESUME FROM EXISTING CHECKPOINT (if present)
# ---------------------------------------------------------------------------

history: list = []
best_val_loss = float("inf")
best_epoch = 0
start_epoch = 1
epochs_without_improvement = 0

if HISTORY_PATH.exists():
    history = pd.read_csv(HISTORY_PATH).to_dict("records")
    print(f"\nResuming: loaded existing history ({len(history)} epochs).")

if BEST_MODEL_PATH.exists():
    print(f"Loading existing LSTM checkpoint: {BEST_MODEL_PATH}")
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    best_val_loss = ckpt["best_val_loss"]
    best_epoch    = ckpt["epoch"]
    start_epoch   = best_epoch + 1
    # Recompute patience counter from history.
    for record in history:
        if record["val_loss"] < best_val_loss + 1e-9:
            if record["epoch"] == best_epoch:
                epochs_without_improvement = 0
        elif record["epoch"] > best_epoch:
            epochs_without_improvement += 1
    print(f"  Resumed from epoch {best_epoch}, best_val_loss={best_val_loss:.6f}")
    print(f"  Next epoch: {start_epoch}")


# ---------------------------------------------------------------------------
# IMAGE -> LSTM INPUT CONVERSION
# ---------------------------------------------------------------------------

def images_to_lstm_input(images: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of Mel-spectrogram images to LSTM input sequences.

    Parameters
    ----------
    images : torch.Tensor  shape [batch, C=1, H=400, W=600]

    Returns
    -------
    seq    : torch.Tensor  shape [batch, T=128, F=64]

    Steps:
      1. Interpolate image to (H_new=F=64, W_new=T=128).
         After interpolate: [batch, 1, 64, 128]
      2. Squeeze channel dim -> [batch, freq=64, time=128]
      3. Permute to [batch, time=128, freq=64]  (LSTM expects [batch, seq, feat])
    """
    # Step 1: resize to [batch, 1, F=64, T=128]
    resized = F.interpolate(
        images,
        size=(LSTM_INPUT_SIZE, LSTM_TIME_STEPS),  # (H_new=64, W_new=128)
        mode="bilinear",
        align_corners=False,
    )
    # Step 2: squeeze channel -> [batch, freq=64, time=128]
    squeezed = resized.squeeze(1)

    # Step 3: permute -> [batch, time=128, freq=64]
    seq = squeezed.permute(0, 2, 1)

    return seq   # [batch, T=128, F=64]


# ---------------------------------------------------------------------------
# EPOCH LOOP FUNCTION
# ---------------------------------------------------------------------------

def run_epoch(loader, training: bool):
    model.train() if training else model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels: list = []
    all_preds: list = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Convert image batch -> LSTM sequence batch
            seq = images_to_lstm_input(images)  # [batch, T, F]

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(seq)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                # Gradient clipping prevents exploding gradients in LSTMs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

if start_epoch > NUM_EPOCHS:
    print(f"\nTraining already completed ({NUM_EPOCHS} epochs). Skipping to evaluation.")
else:
    print(f"\nStarting LSTM training from epoch {start_epoch} / {NUM_EPOCHS} ...")
    training_start = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = run_epoch(train_loader, training=True)
        val_loss,   val_acc   = run_epoch(validation_loader, training=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch":          epoch,
            "train_loss":     train_loss,
            "train_accuracy": train_acc,
            "val_loss":       val_loss,
            "val_accuracy":   val_acc,
            "learning_rate":  current_lr,
        }
        history.append(record)

        # Save history after every epoch (safe even if interrupted).
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch    = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss":       best_val_loss,
                    "lstm_input_size":     LSTM_INPUT_SIZE,
                    "lstm_time_steps":     LSTM_TIME_STEPS,
                    "lstm_hidden":         LSTM_HIDDEN,
                    "lstm_layers":         LSTM_LAYERS,
                    "class_to_label":      {"real": 0, "fake": 1},
                },
                BEST_MODEL_PATH,
            )
            flag = " -> best"
        else:
            epochs_without_improvement += 1
            flag = f" (patience {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE})"

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
            + flag
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after epoch {epoch}.")
            break

    print(f"\nTraining complete. Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")


# ---------------------------------------------------------------------------
# TEST EVALUATION - load best checkpoint
# ---------------------------------------------------------------------------

print("\nLoading best LSTM checkpoint for test evaluation ...")

if not BEST_MODEL_PATH.exists():
    print("ERROR: No LSTM checkpoint found. Training may have been skipped or failed.")
    sys.exit(1)

ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"  Checkpoint epoch  : {ckpt['epoch']}")
print(f"  Best val loss     : {ckpt['best_val_loss']:.6f}")

all_labels_test: list = []
all_preds_test: list = []
all_fake_probs: list = []

torch.manual_seed(SEED)
timing_start = time.perf_counter()

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing LSTM"):
        images = images.to(DEVICE)
        seq = images_to_lstm_input(images)   # [batch, T, F]

        logits = model(seq)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        all_labels_test.extend(labels.tolist())
        all_preds_test.extend(preds.cpu().tolist())
        all_fake_probs.extend(probs[:, 1].cpu().tolist())

timing_end = time.perf_counter()
total_inference_time = timing_end - timing_start
per_sample_ms = (total_inference_time / len(all_labels_test)) * 1000.0

all_labels_test = np.array(all_labels_test)
all_preds_test  = np.array(all_preds_test)
all_fake_probs  = np.array(all_fake_probs)

assert len(all_labels_test) == len(test_dataset), "Test sample count mismatch!"

# Model size
model_size_mb = BEST_MODEL_PATH.stat().st_size / (1024 ** 2)


# ---------------------------------------------------------------------------
# METRICS  (positive class = FAKE = 1)
# ---------------------------------------------------------------------------

accuracy  = accuracy_score(all_labels_test, all_preds_test)

precision = precision_score(all_labels_test, all_preds_test, pos_label=1, zero_division=0)
recall    = recall_score(   all_labels_test, all_preds_test, pos_label=1, zero_division=0)
f1        = f1_score(       all_labels_test, all_preds_test, pos_label=1, zero_division=0)

precision_macro  = precision_score(all_labels_test, all_preds_test, average="macro",    zero_division=0)
recall_macro     = recall_score(   all_labels_test, all_preds_test, average="macro",    zero_division=0)
f1_macro         = f1_score(       all_labels_test, all_preds_test, average="macro",    zero_division=0)

precision_weighted = precision_score(all_labels_test, all_preds_test, average="weighted", zero_division=0)
recall_weighted    = recall_score(   all_labels_test, all_preds_test, average="weighted", zero_division=0)
f1_weighted        = f1_score(       all_labels_test, all_preds_test, average="weighted", zero_division=0)

try:
    auc = roc_auc_score(all_labels_test, all_fake_probs)
except ValueError:
    auc = float("nan")

cm = confusion_matrix(all_labels_test, all_preds_test, labels=[0, 1])

report = classification_report(
    all_labels_test,
    all_preds_test,
    labels=[0, 1],
    target_names=["REAL", "FAKE"],
    zero_division=0,
)


# ---------------------------------------------------------------------------
# PRINT AND SAVE
# ---------------------------------------------------------------------------

summary = (
    "=" * 60 + "\n"
    "  LSTM - TEST SET METRICS\n"
    "=" * 60 + "\n\n"
    f"Checkpoint          : {BEST_MODEL_PATH.name}\n"
    f"Checkpoint epoch    : {ckpt['epoch']}\n"
    f"Best val loss       : {ckpt['best_val_loss']:.6f}\n"
    f"Test samples        : {len(all_labels_test)}\n"
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
    f"Per-sample time     : {per_sample_ms:.3f} ms\n"
    f"Checkpoint size     : {model_size_mb:.1f} MB\n\n"
    "--- Confusion matrix (rows=actual, cols=predicted) ---\n"
    f"                Pred REAL   Pred FAKE\n"
    f"  Actual REAL : {cm[0,0]:>9}   {cm[0,1]:>9}\n"
    f"  Actual FAKE : {cm[1,0]:>9}   {cm[1,1]:>9}\n\n"
    "--- Full classification report ---\n"
    + report
)

print("\n" + summary)

METRICS_PATH.write_text(summary, encoding="utf-8")
print(f"Metrics saved to   : {METRICS_PATH}")

pd.DataFrame(
    cm,
    index=["Actual_REAL", "Actual_FAKE"],
    columns=["Predicted_REAL", "Predicted_FAKE"],
).to_csv(CONFUSION_PATH)
print(f"Confusion matrix   : {CONFUSION_PATH}")

np.savez(
    NUMPY_OUT,
    labels=all_labels_test,
    predictions=all_preds_test,
    fake_probabilities=all_fake_probs,
    inference_time_total_s=np.float64(total_inference_time),
    inference_time_per_sample_ms=np.float64(per_sample_ms),
    model_size_mb=np.float64(model_size_mb),
)
print(f"Arrays saved to    : {NUMPY_OUT}")

print("\n" + "=" * 60)
print("  LSTM TRAINING AND EVALUATION COMPLETE")
print("=" * 60)
