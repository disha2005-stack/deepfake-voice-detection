"""
compare_models.py
-----------------
Loads saved evaluation arrays from both ResNet18 and LSTM, then generates:

  1. Comparison table (CSV)
  2. Confusion matrix plots (one per model)
  3. Bar charts: accuracy, precision, recall, F1-score
  4. Training / validation loss curves (one per model)

Inputs expected (produced by evaluate_resnet18.py and train_lstm.py):
  results/resnet18_test_arrays.npz
  results/lstm_test_arrays.npz
  training/runs/resnet18_history.csv
  training/runs/lstm_history.csv

Outputs (all in results/):
  results/comparison_table.csv
  results/resnet18_confusion_matrix.png
  results/lstm_confusion_matrix.png
  results/accuracy_comparison.png
  results/precision_comparison.png
  results/recall_comparison.png
  results/f1_comparison.png
  results/loss_curves_resnet18.png
  results/loss_curves_lstm.png

Fair comparison guarantee:
  Both models are evaluated on IDENTICAL test samples (same labels array in
  both .npz files, both produced from the same test_loader using the same
  split CSV). A sanity check confirms the label arrays match before plotting.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend - no GUI required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = Path(__file__).resolve().parent   # results/
TRAINING_DIR = PROJECT_ROOT / "training"

RN_NPZ  = RESULTS_DIR / "resnet18_test_arrays.npz"
LST_NPZ = RESULTS_DIR / "lstm_test_arrays.npz"
RN_HIST = TRAINING_DIR / "runs" / "resnet18_history.csv"
LST_HIST = TRAINING_DIR / "runs" / "lstm_history.csv"


# ---------------------------------------------------------------------------
# LOAD ARRAYS
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  MODEL COMPARISON REPORT")
print("=" * 60)

for path in (RN_NPZ, LST_NPZ):
    if not path.exists():
        print(f"\nERROR: Required file not found: {path}")
        print("Please run evaluate_resnet18.py and train_lstm.py first.")
        sys.exit(1)

rn  = np.load(RN_NPZ,  allow_pickle=False)
lst = np.load(LST_NPZ, allow_pickle=False)

rn_labels = rn["labels"]
rn_preds  = rn["predictions"]
rn_probs  = rn["fake_probabilities"]
rn_time_total = float(rn["inference_time_total_s"])
rn_time_ps    = float(rn["inference_time_per_sample_ms"])
rn_size_mb    = float(rn["model_size_mb"])

lst_labels = lst["labels"]
lst_preds  = lst["predictions"]
lst_probs  = lst["fake_probabilities"]
lst_time_total = float(lst["inference_time_total_s"])
lst_time_ps    = float(lst["inference_time_per_sample_ms"])
lst_size_mb    = float(lst["model_size_mb"])

# ---------------------------------------------------------------------------
# FAIR COMPARISON SANITY CHECK
# Both label arrays must be identical (same test samples).
# ---------------------------------------------------------------------------

assert len(rn_labels) == len(lst_labels), (
    f"Label count mismatch: ResNet18 has {len(rn_labels)} samples, "
    f"LSTM has {len(lst_labels)} samples."
)

if not np.array_equal(rn_labels, lst_labels):
    print("WARNING: ResNet18 and LSTM label arrays differ!")
    print("This means they may NOT have been evaluated on the same test samples.")
    print("Results below may NOT be a fair comparison.")
else:
    print(f"\nFair comparison confirmed: both models evaluated on {len(rn_labels)} identical test samples. OK")

n_test = len(rn_labels)


# ---------------------------------------------------------------------------
# COMPUTE METRICS (positive class = FAKE = 1)
# ---------------------------------------------------------------------------

def compute_metrics(labels, preds, probs, name):
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec  = recall_score(   labels, preds, pos_label=1, zero_division=0)
    f1   = f1_score(       labels, preds, pos_label=1, zero_division=0)

    prec_mac = precision_score(labels, preds, average="macro",    zero_division=0)
    rec_mac  = recall_score(   labels, preds, average="macro",    zero_division=0)
    f1_mac   = f1_score(       labels, preds, average="macro",    zero_division=0)

    prec_wt  = precision_score(labels, preds, average="weighted", zero_division=0)
    rec_wt   = recall_score(   labels, preds, average="weighted", zero_division=0)
    f1_wt    = f1_score(       labels, preds, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    print(f"\n--- {name} metrics ---")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision (FAKE): {prec:.4f}")
    print(f"  Recall    (FAKE): {rec:.4f}")
    print(f"  F1        (FAKE): {f1:.4f}")
    print(f"  Precision (macro)   : {prec_mac:.4f}")
    print(f"  Recall    (macro)   : {rec_mac:.4f}")
    print(f"  F1        (macro)   : {f1_mac:.4f}")
    print(f"  Precision (weighted): {prec_wt:.4f}")
    print(f"  Recall    (weighted): {rec_wt:.4f}")
    print(f"  F1        (weighted): {f1_wt:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

    return dict(
        accuracy=acc,
        precision_fake=prec,
        recall_fake=rec,
        f1_fake=f1,
        precision_macro=prec_mac,
        recall_macro=rec_mac,
        f1_macro=f1_mac,
        precision_weighted=prec_wt,
        recall_weighted=rec_wt,
        f1_weighted=f1_wt,
        roc_auc=auc,
        cm=cm,
    )


rn_metrics  = compute_metrics(rn_labels,  rn_preds,  rn_probs,  "ResNet18")
lst_metrics = compute_metrics(lst_labels, lst_preds, lst_probs, "LSTM")


# ---------------------------------------------------------------------------
# COMPARISON TABLE
# ---------------------------------------------------------------------------

table = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision (FAKE)",
        "Recall (FAKE)",
        "F1-score (FAKE)",
        "Precision (macro)",
        "Recall (macro)",
        "F1-score (macro)",
        "Precision (weighted)",
        "Recall (weighted)",
        "F1-score (weighted)",
        "ROC-AUC",
        "Inference time total (s)",
        "Inference time per sample (ms)",
        "Model size (MB)",
        "Test samples",
    ],
    "ResNet18": [
        f"{rn_metrics['accuracy']:.4f}",
        f"{rn_metrics['precision_fake']:.4f}",
        f"{rn_metrics['recall_fake']:.4f}",
        f"{rn_metrics['f1_fake']:.4f}",
        f"{rn_metrics['precision_macro']:.4f}",
        f"{rn_metrics['recall_macro']:.4f}",
        f"{rn_metrics['f1_macro']:.4f}",
        f"{rn_metrics['precision_weighted']:.4f}",
        f"{rn_metrics['recall_weighted']:.4f}",
        f"{rn_metrics['f1_weighted']:.4f}",
        f"{rn_metrics['roc_auc']:.4f}",
        f"{rn_time_total:.2f}",
        f"{rn_time_ps:.3f}",
        f"{rn_size_mb:.1f}",
        str(n_test),
    ],
    "LSTM": [
        f"{lst_metrics['accuracy']:.4f}",
        f"{lst_metrics['precision_fake']:.4f}",
        f"{lst_metrics['recall_fake']:.4f}",
        f"{lst_metrics['f1_fake']:.4f}",
        f"{lst_metrics['precision_macro']:.4f}",
        f"{lst_metrics['recall_macro']:.4f}",
        f"{lst_metrics['f1_macro']:.4f}",
        f"{lst_metrics['precision_weighted']:.4f}",
        f"{lst_metrics['recall_weighted']:.4f}",
        f"{lst_metrics['f1_weighted']:.4f}",
        f"{lst_metrics['roc_auc']:.4f}",
        f"{lst_time_total:.2f}",
        f"{lst_time_ps:.3f}",
        f"{lst_size_mb:.1f}",
        str(n_test),
    ],
})

table_path = RESULTS_DIR / "comparison_table.csv"
table.to_csv(table_path, index=False)
print(f"\nComparison table saved: {table_path}")
print("\n" + table.to_string(index=False))


# ---------------------------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------------------------

COLORS = {
    "ResNet18": "#4C72B0",
    "LSTM":     "#DD8452",
}


def save_confusion_matrix(cm: np.ndarray, model_name: str, filename: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["REAL", "FAKE"], fontsize=12)
    ax.set_yticklabels(["REAL", "FAKE"], fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"{model_name} - Confusion Matrix\n(test set, n={n_test})", fontsize=13)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    plt.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_bar_chart(metric_name: str, rn_val: float, lst_val: float, filename: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    models = ["ResNet18", "LSTM"]
    values = [rn_val, lst_val]
    bars = ax.bar(models, values, color=[COLORS["ResNet18"], COLORS["LSTM"]], width=0.45)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} Comparison\n(test set, n={n_test})", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_loss_curves(history_path: Path, model_name: str, filename: str):
    if not history_path.exists():
        print(f"WARNING: History not found at {history_path} - skipping loss curve.")
        return

    hist = pd.read_csv(history_path)
    epochs = hist["epoch"].tolist()
    train_loss = hist["train_loss"].tolist()
    val_loss   = hist["val_loss"].tolist()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, "o-", color="#4C72B0", label="Train loss", linewidth=2, markersize=4)
    ax.plot(epochs, val_loss,   "s-", color="#DD8452", label="Val loss",   linewidth=2, markersize=4)

    # Mark best epoch (min val loss).
    best_idx = int(np.argmin(val_loss))
    ax.axvline(x=epochs[best_idx], color="green", linestyle="--", linewidth=1.2, label=f"Best (epoch {epochs[best_idx]})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"{model_name} - Training vs. Validation Loss", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Only show integer ticks for epochs.
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# GENERATE ALL PLOTS
# ---------------------------------------------------------------------------

print("\n--- Generating plots ---")

save_confusion_matrix(rn_metrics["cm"],  "ResNet18", "resnet18_confusion_matrix.png")
save_confusion_matrix(lst_metrics["cm"], "LSTM",     "lstm_confusion_matrix.png")

save_bar_chart("Accuracy",         rn_metrics["accuracy"],      lst_metrics["accuracy"],      "accuracy_comparison.png")
save_bar_chart("Precision (FAKE)", rn_metrics["precision_fake"], lst_metrics["precision_fake"], "precision_comparison.png")
save_bar_chart("Recall (FAKE)",    rn_metrics["recall_fake"],    lst_metrics["recall_fake"],    "recall_comparison.png")
save_bar_chart("F1-score (FAKE)",  rn_metrics["f1_fake"],        lst_metrics["f1_fake"],        "f1_comparison.png")

save_loss_curves(RN_HIST,  "ResNet18", "loss_curves_resnet18.png")
save_loss_curves(LST_HIST, "LSTM",     "loss_curves_lstm.png")

print("\n" + "=" * 60)
print("  COMPARISON COMPLETE - all outputs in results/")
print("=" * 60)
