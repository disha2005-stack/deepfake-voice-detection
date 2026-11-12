
from pathlib import Path
import copy
import random
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

from dataset_loader import (
    train_loader,
    validation_loader,
    test_loader,
)
from train_config import (
    CLASS_WEIGHTS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    DEVICE,
)
from resnet18_model import create_resnet18


# ============================================================
# REPRODUCIBILITY AND PATHS
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "training" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "training" / "runs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "resnet18_best.pth"
HISTORY_PATH = RESULTS_DIR / "resnet18_history.csv"
METRICS_PATH = RESULTS_DIR / "resnet18_test_metrics.txt"
CONFUSION_PATH = RESULTS_DIR / "resnet18_confusion_matrix.csv"

# CPU training on 400x600 inputs is expensive.
# Preserve aspect ratio while reducing computation.
INPUT_SIZE = (224, 336)

# Stop if validation loss does not improve for this many epochs.
EARLY_STOPPING_PATIENCE = 6


# ============================================================
# DATA AND MODEL
# ============================================================

print("\nDevice:", DEVICE)
print("Training samples:", len(train_loader.dataset))
print("Validation samples:", len(validation_loader.dataset))
print("Test samples:", len(test_loader.dataset))
print("Input size:", INPUT_SIZE)

# Downloads pretrained weights on the first run if not cached.
model = create_resnet18(pretrained=True).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss(
    weight=CLASS_WEIGHTS.to(DEVICE)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)

# ============================================================
# RESUME TRAINING FROM THE BEST SAVED CHECKPOINT
# ============================================================

history = []

if HISTORY_PATH.exists():
    history = pd.read_csv(HISTORY_PATH).to_dict("records")

best_val_loss = float("inf")
best_epoch = 0
start_epoch = 1
epochs_without_improvement = 0

if BEST_MODEL_PATH.exists():
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_val_loss = checkpoint["best_val_loss"]
    best_epoch = checkpoint["epoch"]
    start_epoch = best_epoch + 1

    print(f"Resuming from saved epoch {best_epoch}")
    print(f"Next epoch: {start_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")

# ============================================================
# TRAINING AND VALIDATION
# ============================================================

def run_epoch(loader, training):
    model.train() if training else model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Resize tensors without modifying the dataset files.
            images = F.interpolate(
                images,
                size=INPUT_SIZE,
                mode="bilinear",
                align_corners=False,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


print("\nStarting ResNet18 training...")

start_time = time.time()

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    epoch_start = time.time()

    train_loss, train_accuracy = run_epoch(
        train_loader, training=True
    )

    val_loss, val_accuracy = run_epoch(
        validation_loader, training=False
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr,
    }
    history.append(record)

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    improved = val_loss < best_val_loss

    if improved:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        # Save the best model, not just the final epoch.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "input_size": INPUT_SIZE,
                "class_to_label": {"real": 0, "fake": 1},
            },
            BEST_MODEL_PATH,
        )
    else:
        epochs_without_improvement += 1

    elapsed = time.time() - epoch_start

    print(
        f"\nEpoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_accuracy:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_accuracy:.4f} | "
        f"LR: {current_lr:.6g} | "
        f"Time: {elapsed:.1f}s"
    )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping: validation loss did not improve.")
        break


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nLoading the best validation checkpoint...")

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_labels = []
all_predictions = []
all_fake_probabilities = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(DEVICE)
        images = F.interpolate(
            images,
            size=INPUT_SIZE,
            mode="bilinear",
            align_corners=False,
        )

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_fake_probabilities.extend(
            probabilities[:, 1].cpu().tolist()
        )

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(
    all_labels, all_predictions, zero_division=0
)
recall = recall_score(
    all_labels, all_predictions, zero_division=0
)
f1 = f1_score(
    all_labels, all_predictions, zero_division=0
)

try:
    auc = roc_auc_score(all_labels, all_fake_probabilities)
except ValueError:
    auc = float("nan")

cm = confusion_matrix(
    all_labels, all_predictions, labels=[0, 1]
)

pd.DataFrame(
    cm,
    index=["Actual_REAL", "Actual_FAKE"],
    columns=["Predicted_REAL", "Predicted_FAKE"],
).to_csv(CONFUSION_PATH)

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=["REAL", "FAKE"],
    zero_division=0,
)

total_time = time.time() - start_time

summary = (
    f"Best epoch: {best_epoch}\n"
    f"Best validation loss: {best_val_loss:.4f}\n"
    f"Test accuracy: {accuracy:.4f}\n"
    f"Test precision (FAKE): {precision:.4f}\n"
    f"Test recall (FAKE): {recall:.4f}\n"
    f"Test F1-score (FAKE): {f1:.4f}\n"
    f"Test ROC-AUC: {auc:.4f}\n"
    f"Total training and evaluation time: {total_time:.1f}s\n\n"
    f"Confusion matrix (rows=actual, columns=predicted):\n{cm}\n\n"
    f"Classification report:\n{report}"
)

METRICS_PATH.write_text(summary, encoding="utf-8")

print("\n" + "=" * 55)
print("RESNET18 TEST RESULTS")
print("=" * 55)
print(summary)
print("\nSaved best model:", BEST_MODEL_PATH)
print("Saved history:", HISTORY_PATH)
print("Saved metrics:", METRICS_PATH)
print("Saved confusion matrix:", CONFUSION_PATH)
