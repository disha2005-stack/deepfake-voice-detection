from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\PBL 5")

INPUT_FILE = next(
    (PROJECT_ROOT / "dataset" / "cleaned" / "real").glob("*.wav")
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "features"
    / "mel_spectrograms"
    / "real"
    / "test_mel.png"
)

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# LOAD CLEANED AUDIO
# ============================================================

print("=" * 60)
print("MEL-SPECTROGRAM TEST")
print("=" * 60)

print(f"\nInput audio:")
print(INPUT_FILE)

audio, sr = librosa.load(
    INPUT_FILE,
    sr=SAMPLE_RATE,
    mono=True
)

print(f"\nSample rate: {sr} Hz")
print(f"Samples: {len(audio)}")
print(f"Duration: {len(audio) / sr:.2f} seconds")


# ============================================================
# CREATE MEL-SPECTROGRAM
# ============================================================

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
    ref=1.0
)


# ============================================================
# SAVE SPECTROGRAM IMAGE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(
    format="%+2.0f dB"
)

plt.title("Mel-Spectrogram - Real Voice")

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=150
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("\nMel-spectrogram created successfully!")

print(f"\nSaved at:")
print(OUTPUT_FILE)

print("\nMel-spectrogram shape:")
print(mel.shape)