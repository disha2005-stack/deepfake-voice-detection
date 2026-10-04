import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

PROJECT = r"D:\PBL 5"

INPUT_DIR = os.path.join(PROJECT, "dataset", "cleaned")
OUTPUT_DIR = os.path.join(PROJECT, "features", "mel_spectrograms")

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def generate_mels(class_name):

    input_dir = os.path.join(INPUT_DIR, class_name)
    output_dir = os.path.join(OUTPUT_DIR, class_name)

    os.makedirs(output_dir, exist_ok=True)

    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".wav")
    ]

    total = len(files)
    success = 0
    failed = 0

    print(f"\n========== {class_name.upper()} ==========")
    print(f"Found {total} WAV files")

    for index, filename in enumerate(files, start=1):

        input_path = os.path.join(input_dir, filename)

        output_name = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, output_name)

        try:

            # Load normalized audio
            audio, sr = librosa.load(
                input_path,
                sr=SAMPLE_RATE,
                mono=True
            )

            # Generate Mel-spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )

            # Convert power to decibels
            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            # Save spectrogram
            plt.figure(figsize=(4, 4))

            librosa.display.specshow(
                mel_db,
                sr=sr,
                hop_length=HOP_LENGTH,
                x_axis=None,
                y_axis=None
            )

            plt.axis("off")
            plt.tight_layout(pad=0)

            plt.savefig(
                output_path,
                bbox_inches="tight",
                pad_inches=0
            )

            plt.close()

            success += 1

            if index % 100 == 0 or index == total:
                print(
                    f"[{index}/{total}] "
                    f"Success: {success} | Failed: {failed}"
                )

        except Exception as e:

            failed += 1

            print(f"[{index}/{total}] FAILED: {filename}")
            print(f"Reason: {e}")

    print(f"\n{class_name.upper()} COMPLETE")
    print(f"Success: {success}")
    print(f"Failed : {failed}")

    return success, failed


print("==========================================")
print("       MEL-SPECTROGRAM GENERATION")
print("==========================================")
print(f"Sample rate : {SAMPLE_RATE} Hz")
print(f"Mel bands   : {N_MELS}")
print(f"FFT size    : {N_FFT}")
print(f"Hop length  : {HOP_LENGTH}")
print("==========================================")

real_result = generate_mels("real")
fake_result = generate_mels("fake")

print("\n==========================================")
print("             FINAL SUMMARY")
print("==========================================")

print(
    f"REAL → Success: {real_result[0]}, "
    f"Failed: {real_result[1]}"
)

print(
    f"FAKE → Success: {fake_result[0]}, "
    f"Failed: {fake_result[1]}"
)

print("\nMel-spectrogram generation complete.")