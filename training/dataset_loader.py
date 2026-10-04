from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_ROOT / "dataset" / "splits" / "indian_dataset_split.csv"
MEL_ROOT = PROJECT_ROOT / "features" / "indian" / "mel_spectrograms"


# ============================================================
# LABELS
# ============================================================

CLASS_TO_LABEL = {
    "real": 0,
    "fake": 1
}


# ============================================================
# SPECTROGRAM AUGMENTATION
# ============================================================

class SpecAugment:

    def __init__(
        self,
        time_mask_width=30,
        frequency_mask_width=12,
        probability=0.5
    ):
        self.time_mask_width = time_mask_width
        self.frequency_mask_width = frequency_mask_width
        self.probability = probability

    def __call__(self, image):

        if torch.rand(1).item() > self.probability:
            return image

        # image shape: [1, frequency, time]
        _, frequency, time = image.shape

        # -------------------------
        # Frequency masking
        # -------------------------
        freq_width = torch.randint(
            0,
            self.frequency_mask_width + 1,
            (1,)
        ).item()

        if freq_width > 0 and freq_width < frequency:

            freq_start = torch.randint(
                0,
                frequency - freq_width + 1,
                (1,)
            ).item()

            image[
                :,
                freq_start:freq_start + freq_width,
                :
            ] = 0

        # -------------------------
        # Time masking
        # -------------------------
        time_width = torch.randint(
            0,
            self.time_mask_width + 1,
            (1,)
        ).item()

        if time_width > 0 and time_width < time:

            time_start = torch.randint(
                0,
                time - time_width + 1,
                (1,)
            ).item()

            image[
                :,
                :,
                time_start:time_start + time_width
            ] = 0

        return image

# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.197581],
        std=[0.200545]
    ),
    SpecAugment(
        time_mask_width=30,
        frequency_mask_width=12,
        probability=0.5
    )
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.197581],
        std=[0.200545]
    )
])

# ============================================================
# DATASET
# ============================================================

class IndianMelDataset(Dataset):

    def __init__(self, csv_file, mel_root, split, transform=None):

        self.data = pd.read_csv(csv_file)

        # Select requested split
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)

        self.mel_root = Path(mel_root)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        filename = row["filename"]
        class_name = row["class"]
        language = row["language"]

        image_path = (
            self.mel_root
            / class_name
            / language
            / filename
        )

        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        label = CLASS_TO_LABEL[class_name]

        return image, label


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = IndianMelDataset(
    CSV_FILE,
    MEL_ROOT,
    "train",
    train_transform
)

validation_dataset = IndianMelDataset(
    CSV_FILE,
    MEL_ROOT,
    "validation",
    eval_transform
)

test_dataset = IndianMelDataset(
    CSV_FILE,
    MEL_ROOT,
    "test",
    eval_transform
)

# ============================================================
# CREATE DATALOADERS
# ============================================================

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# VERIFICATION
# ============================================================

print("\n===================================================")
print("          DATASET LOADER VERIFICATION")
print("===================================================")

print(f"Train samples      : {len(train_dataset)}")
print(f"Validation samples : {len(validation_dataset)}")
print(f"Test samples       : {len(test_dataset)}")

images, labels = next(iter(train_loader))

print(f"\nBatch image shape  : {images.shape}")
print(f"Batch label shape  : {labels.shape}")
print(f"Image dtype        : {images.dtype}")
print(f"Label dtype        : {labels.dtype}")
print(f"Image min          : {images.min().item():.4f}")
print(f"Image max          : {images.max().item():.4f}")

print("\nFirst 10 labels:")
print(labels[:10].tolist())

print("\n===================================================")
print("RESULT: DATASET LOADER TEST COMPLETED")
print("===================================================\n")