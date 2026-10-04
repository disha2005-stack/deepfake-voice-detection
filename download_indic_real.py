from datasets import load_dataset
from pathlib import Path
import csv
import time

# ============================================================
# SETTINGS
# ============================================================

LANGUAGES = {
    "kannada": "kn",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te"
}

SAMPLES_PER_LANGUAGE = 500

BASE_DIR = Path("dataset/raw/real")


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_language(language_name, language_code):

    output_dir = BASE_DIR / language_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = output_dir / "metadata.csv"

    print("\n" + "=" * 60)
    print(f"Processing: {language_name.upper()}")
    print("=" * 60)

    print("Loading IndicVoices...")

    dataset = load_dataset(
        "ai4bharat/IndicVoices",
        language_name,
        split="train",
        streaming=True
    )

    # Disable automatic audio decoding.
    # We only need the raw audio bytes.
    dataset = dataset.cast_column(
        "audio_filepath",
        dataset.features["audio_filepath"].__class__(
            decode=False
        )
    )

    print("Dataset ready.")
    print(f"Target: {SAMPLES_PER_LANGUAGE} samples\n")

    # --------------------------------------------------------
    # Existing files
    # --------------------------------------------------------

    existing_files = list(output_dir.glob(f"{language_name}_*.flac"))

    count = len(existing_files)

    if count >= SAMPLES_PER_LANGUAGE:
        print(
            f"{language_name}: already has "
            f"{count} samples. Skipping."
        )
        return

    # --------------------------------------------------------
    # Load existing speaker IDs from metadata
    # --------------------------------------------------------

    seen_speakers = set()

    if metadata_file.exists():

        with open(
            metadata_file,
            "r",
            encoding="utf-8",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                if row.get("speaker_id"):
                    seen_speakers.add(row["speaker_id"])

    # --------------------------------------------------------
    # Create metadata file
    # --------------------------------------------------------

    file_exists = metadata_file.exists()

    with open(
        metadata_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.writer(csvfile)

        if not file_exists:

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

        # ----------------------------------------------------
        # Stream dataset
        # ----------------------------------------------------

        for sample in dataset:

            audio = sample["audio_filepath"]

            audio_bytes = audio.get("bytes")

            if audio_bytes is None:
                print("Skipping sample: audio bytes missing.")
                continue

            speaker_id = sample["speaker_id"]

            # -----------------------------------------------
            # One recording per speaker
            # -----------------------------------------------

            if speaker_id in seen_speakers:
                continue

            filename = (
                f"{language_name}_{count + 1:03d}.flac"
            )

            output_path = output_dir / filename

            # -----------------------------------------------
            # Save audio
            # -----------------------------------------------

            try:

                with open(output_path, "wb") as audio_file:
                    audio_file.write(audio_bytes)

            except Exception as e:

                print(
                    f"Error saving {filename}: {e}"
                )

                continue

            # -----------------------------------------------
            # Save metadata
            # -----------------------------------------------

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

            csvfile.flush()

            seen_speakers.add(speaker_id)

            count += 1

            print(
                f"[{count}/{SAMPLES_PER_LANGUAGE}] "
                f"{filename} | "
                f"{sample['duration']:.2f}s | "
                f"Speaker: {speaker_id}"
            )

            if count >= SAMPLES_PER_LANGUAGE:
                break

            # Small pause to avoid hammering the server
            time.sleep(0.05)

    print("\nCompleted:", language_name)
    print("Samples:", count)
    print("Folder:", output_dir)
    print("Metadata:", metadata_file)


# ============================================================
# MAIN PROGRAM
# ============================================================

print("\nINDICVOICES INDIAN LANGUAGE DATA COLLECTION")
print("=" * 60)

for language, code in LANGUAGES.items():

    try:

        download_language(
            language,
            code
        )

    except Exception as e:

        print("\nERROR processing", language)
        print(e)

        print(
            "\nThe program will continue "
            "with the next language."
        )

print("\n" + "=" * 60)
print("ALL LANGUAGE PROCESSING FINISHED")
print("=" * 60)