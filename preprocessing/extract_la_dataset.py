import os
import subprocess

ARCHIVE = r"D:\PBL 5\LA.zip"

REAL_DIR = r"D:\PBL 5\dataset\raw\real"
FAKE_DIR = r"D:\PBL 5\dataset\raw\fake"

PROTOCOL = (
    "LA/ASVspoof2019_LA_cm_protocols/"
    "ASVspoof2019.LA.cm.train.trn.txt"
)

TARGET_PER_CLASS = 2580

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)


def extract_file(archive_path, member_path, output_path):
    with open(output_path, "wb") as output:
        result = subprocess.run(
            ["tar", "-xOf", archive_path, member_path],
            stdout=output,
            stderr=subprocess.PIPE
        )

    if result.returncode != 0:
        os.remove(output_path)
        raise RuntimeError(
            f"Failed to extract {member_path}\n"
            f"{result.stderr.decode(errors='ignore')}"
        )


print("Reading protocol...")

protocol_result = subprocess.run(
    ["tar", "-xOf", ARCHIVE, PROTOCOL],
    capture_output=True
)

if protocol_result.returncode != 0:
    raise RuntimeError(
        "Could not read the protocol file.\n"
        + protocol_result.stderr.decode(errors="ignore")
    )

lines = protocol_result.stdout.decode("utf-8").splitlines()

real_count = 0
fake_count = 0

print("Starting extraction...\n")

for line in lines:

    parts = line.split()

    if len(parts) < 5:
        continue

    file_id = parts[1]
    label = parts[4]

    audio_path = (
        "LA/ASVspoof2019_LA_train/flac/"
        f"{file_id}.flac"
    )

    if label == "bonafide" and real_count < TARGET_PER_CLASS:

        output_path = os.path.join(
            REAL_DIR,
            f"{file_id}.flac"
        )

        extract_file(ARCHIVE, audio_path, output_path)

        real_count += 1
        print(f"REAL: {real_count}/{TARGET_PER_CLASS}")

    elif label == "spoof" and fake_count < TARGET_PER_CLASS:

        output_path = os.path.join(
            FAKE_DIR,
            f"{file_id}.flac"
        )

        extract_file(ARCHIVE, audio_path, output_path)

        fake_count += 1
        print(f"FAKE: {fake_count}/{TARGET_PER_CLASS}")

    if (
        real_count == TARGET_PER_CLASS
        and fake_count == TARGET_PER_CLASS
    ):
        break


print("\n========== EXTRACTION COMPLETE ==========")
print(f"REAL files: {real_count}")
print(f"FAKE files: {fake_count}")
print(f"TOTAL files: {real_count + fake_count}")