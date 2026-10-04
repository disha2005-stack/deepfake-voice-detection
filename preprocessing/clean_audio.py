import os
import librosa
import soundfile as sf
import numpy as np

PROJECT = r"D:\PBL 5"

RAW_DIR = os.path.join(PROJECT, "dataset", "raw")
CLEANED_DIR = os.path.join(PROJECT, "dataset", "cleaned")

TARGET_SR = 16000
MIN_DURATION = 1.0
MAX_DURATION = 15.0


def process_class(class_name):

    input_dir = os.path.join(RAW_DIR, class_name)
    output_dir = os.path.join(CLEANED_DIR, class_name)

    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".flac")
    ]

    total = len(files)
    success = 0
    failed = 0
    skipped = 0

    print(f"\n========== {class_name.upper()} ==========")
    print(f"Found {total} FLAC files")

    for index, filename in enumerate(files, start=1):

        input_path = os.path.join(input_dir, filename)

        output_name = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_dir, output_name)

        try:

            # Load audio and convert to mono + 16 kHz
            audio, sr = librosa.load(
                input_path,
                sr=TARGET_SR,
                mono=True
            )

            # Check that audio contains samples
            if len(audio) == 0:
                skipped += 1
                print(f"[{index}/{total}] SKIPPED - empty: {filename}")
                continue

            duration = len(audio) / TARGET_SR

            # Check duration
            if duration < MIN_DURATION or duration > MAX_DURATION:
                skipped += 1
                print(
                    f"[{index}/{total}] SKIPPED - "
                    f"duration {duration:.2f}s: {filename}"
                )
                continue

            # Check for invalid numerical values
            if not np.isfinite(audio).all():
                skipped += 1
                print(f"[{index}/{total}] SKIPPED - invalid audio: {filename}")
                continue

            # Normalize amplitude
            max_value = np.max(np.abs(audio))

            if max_value > 0:
                audio = audio / max_value

            # Save as WAV
            sf.write(
                output_path,
                audio,
                TARGET_SR,
                subtype="PCM_16"
            )

            success += 1

            if index % 50 == 0 or index == total:
                print(
                    f"[{index}/{total}] "
                    f"Processed | Success: {success} | "
                    f"Skipped: {skipped} | Failed: {failed}"
                )

        except Exception as e:

            failed += 1

            print(
                f"[{index}/{total}] FAILED: {filename}"
            )
            print(f"Reason: {e}")

    print(f"\n{class_name.upper()} COMPLETE")
    print(f"Success : {success}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {failed}")

    return success, skipped, failed


print("==========================================")
print("   ASVSPOOF 2019 LA AUDIO PREPROCESSING")
print("==========================================")
print("Target sample rate : 16000 Hz")
print("Channels           : Mono")
print("Output format      : WAV")
print("Duration range     : 1–15 seconds")
print("==========================================")

real_results = process_class("real")
fake_results = process_class("fake")

print("\n==========================================")
print("             FINAL SUMMARY")
print("==========================================")

print(
    f"REAL → Success: {real_results[0]}, "
    f"Skipped: {real_results[1]}, "
    f"Failed: {real_results[2]}"
)

print(
    f"FAKE → Success: {fake_results[0]}, "
    f"Skipped: {fake_results[1]}, "
    f"Failed: {fake_results[2]}"
)

print("\nCleaning and normalization complete.")