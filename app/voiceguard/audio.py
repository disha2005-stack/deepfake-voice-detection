import io

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as functional
from PIL import Image


SAMPLE_RATE = 16000
SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS
MIN_AUDIO_SAMPLES = SAMPLE_RATE
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
FMIN = 20
FMAX = 8000
IMAGE_SIZE = (224, 336)
NORMALIZE_MEAN = 0.197581
NORMALIZE_STD = 0.200545


class AudioInputError(ValueError):
    pass


def load_audio(path):
    try:
        audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as error:
        raise AudioInputError("The uploaded file is not a readable WAV audio file.") from error

    if audio.size == 0 or audio.shape[0] == 0:
        raise AudioInputError("The audio file is empty.")
    if source_rate <= 0 or not np.isfinite(audio).all():
        raise AudioInputError("The audio contains invalid samples or metadata.")

    mono = audio.mean(axis=1, dtype=np.float32)
    if source_rate != SAMPLE_RATE:
        duration = len(mono) / source_rate
        new_length = int(round(duration * SAMPLE_RATE))
        if new_length <= 0:
            raise AudioInputError("The audio is too short to resample.")
        old_positions = np.linspace(0, len(mono) - 1, num=len(mono))
        new_positions = np.linspace(0, len(mono) - 1, num=new_length)
        mono = np.interp(new_positions, old_positions, mono).astype(np.float32)

    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono = mono / peak

    if len(mono) < MIN_AUDIO_SAMPLES:
        raise AudioInputError("Audio must be at least one second long.")
    return mono.astype(np.float32, copy=False)


def split_audio(audio):
    segments = []
    for start in range(0, len(audio), SEGMENT_SAMPLES):
        segment = audio[start : start + SEGMENT_SAMPLES]
        if len(segment) < SEGMENT_SAMPLES:
            segment = np.pad(segment, (0, SEGMENT_SAMPLES - len(segment)))
        segments.append(segment.astype(np.float32, copy=False))
    return segments


def create_spectrogram(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=mel.max())

    figure = plt.figure(figsize=(6, 4))
    try:
        librosa.display.specshow(
            mel_db,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            fmin=FMIN,
            fmax=FMAX,
        )
        plt.axis("off")
        plt.tight_layout(pad=0)
        rendered = io.BytesIO()
        figure.savefig(rendered, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        rendered.seek(0)
        with Image.open(rendered) as image:
            grayscale = image.convert("L")
            pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
    finally:
        plt.close(figure)

    normalized = (pixels - NORMALIZE_MEAN) / NORMALIZE_STD
    tensor = torch.from_numpy(normalized.copy()).unsqueeze(0).unsqueeze(0)
    tensor = functional.interpolate(
        tensor,
        size=IMAGE_SIZE,
        mode="bilinear",
        align_corners=False,
    )
    return tensor, mel_db


def create_display_spectrogram(mel_db):
    figure, axis = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    image = librosa.display.specshow(
        mel_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=FMIN,
        fmax=FMAX,
        ax=axis,
    )
    axis.set_title("Mel-frequency energy · segment 1", loc="left")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    figure.colorbar(image, ax=axis, format="%+2.0f dB", label="Power (dB)")
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=120)
    plt.close(figure)
    return output.getvalue()