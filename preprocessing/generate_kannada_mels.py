import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

# ===================================================
# SETTINGS
# ===================================================

BASE = Path("dataset/cleaned")
OUTPUT_BASE = Path("features/indian/mel_spectrograms")

LANGUAGES = ["hindi", "tamil", "telugu"]

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

# ===================================================
# PROCESS AUDIO
# ===================================================

def generate_mels(data_class, language):

    input_folder = BASE / data_class / language
    output_folder = OUTPUT_BASE / data_class / language

    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(input_folder.glob("*.wav"))

    print(f"\n{data_class.upper()} - {language.upper()}")
    print(f"Input files: {len(files)}")

    for index, file in enumerate(files, start=1):

        try:
            audio, sr = librosa.load(
                file,
                sr=SAMPLE_RATE,
                mono=True
            )

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                fmin=20,
                fmax=8000
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=mel.max()
            )

            output_file = output_folder / f"{file.stem}.png"

            plt.figure(figsize=(6, 4))

            librosa.display.specshow(
                mel_db,
                sr=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                x_axis="time",
                y_axis="mel",
                fmin=20,
                fmax=8000
            )

            plt.axis("off")
            plt.tight_layout(pad=0)

            plt.savefig(
                output_file,
                dpi=100,
                bbox_inches="tight",
                pad_inches=0
            )

            plt.close()

            if index % 100 == 0 or index == len(files):
                print(f"[{index}/{len(files)}] Generated")

        except Exception as e:
            print(f"ERROR: {file.name} -> {e}")


# ===================================================
# MAIN
# ===================================================

print("\n==============================================")
print(" INDIAN LANGUAGE MEL-SPECTROGRAM GENERATION")
print("==============================================")

for language in LANGUAGES:

    print(f"\n########## {language.upper()} ##########")

    generate_mels("real", language)
    generate_mels("fake", language)

print("\n==============================================")
print(" INDIAN LANGUAGE GENERATION COMPLETE")
print("==============================================")