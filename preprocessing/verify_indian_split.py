from pathlib import Path
import csv

CSV_FILE = Path("dataset/splits/indian_dataset_split.csv")

groups = {}

with open(CSV_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:

        key = (
            row["class"],
            row["language"],
            row["source_id"]
        )

        if key not in groups:
            groups[key] = set()

        groups[key].add(row["split"])


leaked_sources = {
    key: splits
    for key, splits in groups.items()
    if len(splits) > 1
}

print("\n===================================================")
print("          DATASET SPLIT VERIFICATION")
print("===================================================")

print(f"Total manifest rows : {sum(len(s) for s in groups.values())}")
print(f"Unique source groups: {len(groups)}")
print(f"Leaked sources     : {len(leaked_sources)}")

print("\nSplit counts:")

for split in ["train", "validation", "test"]:

    count = 0

    with open(CSV_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["split"] == split:
                count += 1

    print(f"{split:12}: {count}")

print("\n===================================================")

if len(leaked_sources) == 0:
    print("RESULT: NO SOURCE LEAKAGE DETECTED")
else:
    print("RESULT: SOURCE LEAKAGE DETECTED!")

print("===================================================\n")