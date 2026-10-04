from pathlib import Path

import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\PBL 5")

INPUT_ROOT = PROJECT_ROOT / "dataset" / "cleaned"
OUTPUT_ROOT = PROJECT_ROOT / "features" / "mel_spectrograms"

SAMPLE_RATE = 16000

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# CREATE MEL-SPECTROGRAM FOR ONE AUDIO FILE
# ============================================================

def create_mel_spectrogram(input_path, output_path):

    try:

        # ----------------------------------------------------
        # Load cleaned audio
        # ----------------------------------------------------

        audio, sr = librosa.load(
            input_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # ----------------------------------------------------
        # Generate Mel-spectrogram
        # ----------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        # ----------------------------------------------------
        # Convert power to decibels
        # ----------------------------------------------------

        mel_db = librosa.power_to_db(
            mel,
            ref=1.0
        )

        # ----------------------------------------------------
        # Create output folder
        # ----------------------------------------------------

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save spectrogram
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 4)
        )

        librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel"
        )

        plt.axis("off")

        plt.tight_layout(
            pad=0
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

        return True

    except Exception as e:

        print(
            f"\nFailed: {input_path.name} → {e}"
        )

        return False


# ============================================================
# PROCESS ONE CLASS
# ============================================================

def process_class(class_name):

    input_folder = INPUT_ROOT / class_name
    output_folder = OUTPUT_ROOT / class_name

    audio_files = list(
        input_folder.glob("*.wav")
    )

    print("\n" + "=" * 60)
    print(f"PROCESSING MEL-SPECTROGRAMS: {class_name.upper()}")
    print("=" * 60)

    print(
        f"Found {len(audio_files)} WAV files"
    )

    successful = 0
    failed = 0

    for input_path in tqdm(
        audio_files,
        desc=f"{class_name.capitalize()} Mel"
    ):

        # Same filename, but PNG

        output_path = output_folder / (
            input_path.stem + ".png"
        )

        success = create_mel_spectrogram(
            input_path,
            output_path
        )

        if success:

            successful += 1

        else:

            failed += 1

    print(
        f"\nSuccessfully created: {successful}"
    )

    print(
        f"Failed: {failed}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MEL-SPECTROGRAM GENERATION")
    print("=" * 60)

    print(
        f"\nInput : {INPUT_ROOT}"
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    # Process REAL

    process_class("real")

    # Process FAKE

    process_class("fake")

    print("\n" + "=" * 60)
    print("MEL-SPECTROGRAM GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"\nFeatures saved at:\n{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()