import os
import shutil

PROJECT = r"D:\PBL 5"

TEMP_AUDIO = os.path.join(
    PROJECT,
    "temp_la_train",
    "LA",
    "ASVspoof2019_LA_train",
    "flac"
)

REAL_DIR = os.path.join(PROJECT, "dataset", "raw", "real")
FAKE_DIR = os.path.join(PROJECT, "dataset", "raw", "fake")

PROTOCOL = os.path.join(
    PROJECT,
    "protocols",
    "ASVspoof2019.LA.cm.train.trn.txt"
)

TARGET = 2580

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

print("Reading protocol...")

with open(PROTOCOL, "r", encoding="utf-8") as f:
    lines = f.readlines()

real_count = 0
fake_count = 0

for line in lines:

    parts = line.split()

    if len(parts) < 5:
        continue

    file_id = parts[1]
    label = parts[4]

    source = os.path.join(
        TEMP_AUDIO,
        file_id + ".flac"
    )

    if not os.path.exists(source):
        print("Missing:", file_id)
        continue

    if label == "bonafide" and real_count < TARGET:

        destination = os.path.join(
            REAL_DIR,
            file_id + ".flac"
        )

        if not os.path.exists(destination):
            shutil.copy2(source, destination)

        real_count += 1
        print(f"REAL: {real_count}/{TARGET}")

    elif label == "spoof" and fake_count < TARGET:

        destination = os.path.join(
            FAKE_DIR,
            file_id + ".flac"
        )

        if not os.path.exists(destination):
            shutil.copy2(source, destination)

        fake_count += 1
        print(f"FAKE: {fake_count}/{TARGET}")

    if real_count == TARGET and fake_count == TARGET:
        break

print("\n========== COMPLETE ==========")
print("REAL:", real_count)
print("FAKE:", fake_count)
print("TOTAL:", real_count + fake_count)