from pathlib import Path
import re
import random
import csv

# ===================================================
# SETTINGS
# ===================================================

BASE = Path("features/indian/mel_spectrograms")
OUTPUT = Path("dataset/splits/indian_dataset_split.csv")

LANGUAGES = ["kannada", "hindi", "tamil", "telugu"]
CLASSES = ["real", "fake"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ===================================================
# CHECK RATIOS
# ===================================================

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

random.seed(RANDOM_SEED)

# ===================================================
# SOURCE GROUP EXTRACTION
# ===================================================

def get_source_id(filename):
    """
    Example:
    kannada_002_seg003.png
    becomes:
    kannada_002
    """

    return re.sub(r"_seg\d+$", "", filename.stem)


# ===================================================
# MAIN
# ===================================================

rows = []

print("\n===================================================")
print("       CREATING LEAKAGE-SAFE DATASET SPLIT")
print("===================================================")

for data_class in CLASSES:

    for language in LANGUAGES:

        folder = BASE / data_class / language
        files = sorted(folder.glob("*.png"))

        # Group files by source recording
        source_groups = {}

        for file in files:

            source_id = get_source_id(file)

            if source_id not in source_groups:
                source_groups[source_id] = []

            source_groups[source_id].append(file)

        source_ids = sorted(source_groups.keys())

        # Shuffle source groups, NOT individual segments
        random.shuffle(source_ids)

        total_sources = len(source_ids)

        train_count = int(total_sources * TRAIN_RATIO)
        val_count = int(total_sources * VAL_RATIO)

        train_sources = set(source_ids[:train_count])

        val_sources = set(
            source_ids[train_count:train_count + val_count]
        )

        test_sources = set(
            source_ids[train_count + val_count:]
        )

        print(
            f"\n{data_class.upper():5} | "
            f"{language.upper():7}"
        )

        print(f"Total source groups : {total_sources}")
        print(f"Train source groups : {len(train_sources)}")
        print(f"Val source groups   : {len(val_sources)}")
        print(f"Test source groups  : {len(test_sources)}")

        # Assign every segment according to its source
        for source_id, source_files in source_groups.items():

            if source_id in train_sources:
                split = "train"

            elif source_id in val_sources:
                split = "validation"

            elif source_id in test_sources:
                split = "test"

            else:
                raise RuntimeError(
                    f"Source was not assigned: {source_id}"
                )

            for file in source_files:

                rows.append({
                    "filename": file.name,
                    "class": data_class,
                    "language": language,
                    "source_id": source_id,
                    "split": split
                })


# ===================================================
# WRITE CSV
# ===================================================

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(
    OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    fieldnames = [
        "filename",
        "class",
        "language",
        "source_id",
        "split"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


# ===================================================
# FINAL SUMMARY
# ===================================================

print("\n===================================================")
print("              SPLIT MANIFEST CREATED")
print("===================================================")

print(f"Total samples : {len(rows)}")
print(f"Train samples : {sum(r['split'] == 'train' for r in rows)}")
print(f"Val samples   : {sum(r['split'] == 'validation' for r in rows)}")
print(f"Test samples  : {sum(r['split'] == 'test' for r in rows)}")

print("\nCSV location:")
print(OUTPUT)

print("===================================================")