from datasets import load_dataset, Audio
from pathlib import Path
import time

LANGUAGE = "Telugu"
TARGET = 500

OUTPUT_DIR = Path(f"dataset/raw/fake/{LANGUAGE.lower()}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Count files already collected so the script can resume safely
existing = list(OUTPUT_DIR.glob("*.wav"))
count = len(existing)

print(f"Language : {LANGUAGE}")
print(f"Already collected : {count}")
print(f"Target : {TARGET}")

if count >= TARGET:
    print("Target already reached.")
    raise SystemExit

while count < TARGET:
    try:
        print("\nConnecting to IndicSynth...")

        dataset = load_dataset(
            "vdivyasharma/IndicSynth",
            name=LANGUAGE,
            split="train",
            streaming=True
        )

        # Do NOT decode through TorchCodec
        dataset = dataset.cast_column(
            "audio",
            Audio(decode=False)
        )

        skipped = 0

        for example in dataset:

            if count >= TARGET:
                break

            audio = example["audio"]

            if audio is None:
                continue

            audio_bytes = audio.get("bytes")
            original_path = audio.get("path")

            if not audio_bytes:
                print("Skipping sample with no audio bytes.")
                continue

            count += 1

            output_file = OUTPUT_DIR / f"telugu_{count:03d}.wav"

            with open(output_file, "wb") as f:
                f.write(audio_bytes)

            print(
                f"[{count:03d}/{TARGET}] "
                f"Saved {output_file.name} | "
                f"Model: {example.get('Generative Model')} | "
                f"Source: {original_path}"
            )

        if count >= TARGET:
            break

        print("Stream ended before reaching target. Retrying...")

    except Exception as e:
        print("\nConnection/error encountered:")
        print(e)
        print("Already downloaded files are safe.")
        print("Retrying in 5 seconds...")
        time.sleep(5)

print("\n================================")
print("Telugu collection completed!")
print(f"Total files: {count}")
print(f"Location: {OUTPUT_DIR}")
print("================================")