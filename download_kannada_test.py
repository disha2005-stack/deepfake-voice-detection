from datasets import load_dataset
from pathlib import Path
import csv

# -----------------------------
# Settings
# -----------------------------
LANGUAGE = "kannada"
TARGET_SAMPLES = 50

OUTPUT_DIR = Path("dataset/raw/real/kannada")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = OUTPUT_DIR / "metadata.csv"

print("Loading IndicVoices Kannada dataset...")

dataset = load_dataset(
    "ai4bharat/IndicVoices",
    LANGUAGE,
    split="train",
    streaming=True
)

# Disable automatic audio decoding
dataset = dataset.cast_column(
    "audio_filepath",
    dataset.features["audio_filepath"].__class__(
        decode=False
    )
)

print("Dataset ready.")
print(f"Collecting {TARGET_SAMPLES} Kannada samples...\n")

count = 0
seen_speakers = set()

with open(METADATA_FILE, "w", newline="", encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "filename",
        "language",
        "duration",
        "speaker_id",
        "gender",
        "age_group",
        "state",
        "district",
        "scenario",
        "task_name",
        "text"
    ])

    for sample in dataset:

        audio = sample["audio_filepath"]

        audio_bytes = audio.get("bytes")

        if audio_bytes is None:
            print("Skipping sample because audio bytes are missing.")
            continue

        speaker_id = sample["speaker_id"]

        # Keep only one recording from each speaker
        # for this initial test.
        if speaker_id in seen_speakers:
            continue

        seen_speakers.add(speaker_id)

        filename = f"kannada_{count + 1:03d}.flac"
        output_path = OUTPUT_DIR / filename

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        writer.writerow([
            filename,
            sample["lang"],
            sample["duration"],
            speaker_id,
            sample["gender"],
            sample["age_group"],
            sample["state"],
            sample["district"],
            sample["scenario"],
            sample["task_name"],
            sample["text"]
        ])

        count += 1

        print(
            f"[{count}/{TARGET_SAMPLES}] "
            f"{filename} | "
            f"{sample['duration']:.2f}s | "
            f"Speaker: {speaker_id}"
        )

        if count >= TARGET_SAMPLES:
            break

print("\n================================")
print("Download completed!")
print("Samples:", count)
print("Output:", OUTPUT_DIR)
print("Metadata:", METADATA_FILE)
print("================================")