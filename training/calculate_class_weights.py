from pathlib import Path
import pandas as pd
import torch


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "splits"
    / "indian_dataset_split.csv"
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

train_df = df[df["split"] == "train"]


# ============================================================
# COUNT CLASSES
# ============================================================

real_count = (train_df["class"] == "real").sum()
fake_count = (train_df["class"] == "fake").sum()

total = real_count + fake_count
num_classes = 2


# ============================================================
# CLASS WEIGHTS
# ============================================================

real_weight = total / (num_classes * real_count)
fake_weight = total / (num_classes * fake_count)


# ============================================================
# RESULTS
# ============================================================

print("\n===================================================")
print("          CLASS IMBALANCE ANALYSIS")
print("===================================================")

print(f"Training samples : {total}")

print(f"\nREAL samples     : {real_count}")
print(f"FAKE samples     : {fake_count}")

print("\nClass proportions:")
print(f"REAL             : {real_count / total:.4f}")
print(f"FAKE             : {fake_count / total:.4f}")

print("\nCalculated class weights:")
print(f"REAL weight      : {real_weight:.6f}")
print(f"FAKE weight      : {fake_weight:.6f}")

weights = torch.tensor(
    [real_weight, fake_weight],
    dtype=torch.float32
)

print("\nPyTorch weights:")
print(weights)

print("\n===================================================")
print("RESULT: CLASS WEIGHTS CALCULATED")
print("===================================================\n")