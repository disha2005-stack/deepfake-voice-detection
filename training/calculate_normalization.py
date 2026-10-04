from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_ROOT / "dataset" / "splits" / "indian_dataset_split.csv"
MEL_ROOT = PROJECT_ROOT / "features" / "indian" / "mel_spectrograms"


# ============================================================
# LOAD TRAINING MANIFEST
# ============================================================

df = pd.read_csv(CSV_FILE)

train_df = df[df["split"] == "train"].reset_index(drop=True)

print("\n===================================================")
print("       TRAINING DATA NORMALIZATION ANALYSIS")
print("===================================================")

print(f"Training samples : {len(train_df)}")


# ============================================================
# CALCULATE MEAN AND STANDARD DEVIATION
# ============================================================

pixel_sum = 0.0
pixel_squared_sum = 0.0
pixel_count = 0

for index, row in train_df.iterrows():

    image_path = (
        MEL_ROOT
        / row["class"]
        / row["language"]
        / row["filename"]
    )

    image = Image.open(image_path).convert("L")

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    pixel_sum += image_array.sum()
    pixel_squared_sum += np.square(image_array).sum()
    pixel_count += image_array.size

    if (index + 1) % 500 == 0:
        print(f"Processed: {index + 1}/{len(train_df)}")


# ============================================================
# FINAL CALCULATION
# ============================================================

mean = pixel_sum / pixel_count

variance = (pixel_squared_sum / pixel_count) - (mean ** 2)

std = np.sqrt(variance)


# ============================================================
# RESULTS
# ============================================================

print("\n===================================================")
print("                 RESULTS")
print("===================================================")

print(f"Mean : {mean:.6f}")
print(f"Std  : {std:.6f}")

print("\n===================================================")
print("RESULT: NORMALIZATION VALUES CALCULATED")
print("===================================================\n")