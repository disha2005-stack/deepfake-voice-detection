import soundfile as sf
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path("dataset/raw")
OUTPUT_BASE = Path("dataset/cleaned")

# Process only the remaining languages.
LANGUAGES = ["hindi", "tamil", "telugu"]
TYPES = ["real", "fake"]

TARGET_SR = 16000
TARGET_DURATION = 5
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION

MIN_DURATION = 1.0


# ============================================================
# FUNCTIONS
# ============================================================

def normalize_audio(audio):
    """Peak-normalize audio safely."""
    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio


def resample_audio(audio, original_sr, target_sr):
    """Resample audio using linear interpolation."""

    if original_sr == target_sr:
        return audio.astype(np.float32)

    duration = len(audio) / original_sr
    new_length = int(round(duration * target_sr))

    old_positions = np.linspace(
        0,
        len(audio) - 1,
        num=len(audio)
    )

    new_positions = np.linspace(
        0,
        len(audio) - 1,
        num=new_length
    )

    return np.interp(
        new_positions,
        old_positions,
        audio
    ).astype(np.float32)


def create_segments(audio):
    """
    Create fixed 5-second segments.

    Returns:
        segments
        status
    """

    length = len(audio)

    # Less than 1 second → exclude
    if length < TARGET_SR * MIN_DURATION:
        return [], "excluded_short"

    # 1–5 seconds → one padded segment
    if length <= TARGET_SAMPLES:

        padded = np.zeros(
            TARGET_SAMPLES,
            dtype=np.float32
        )

        padded[:length] = audio

        return [padded], "padded"

    # More than 5 seconds → multiple segments
    segments = []

    start = 0

    while start < length:

        segment = audio[start:start + TARGET_SAMPLES]

        if len(segment) == 0:
            break

        # Final incomplete segment → pad it
        if len(segment) < TARGET_SAMPLES:

            padded = np.zeros(
                TARGET_SAMPLES,
                dtype=np.float32
            )

            padded[:len(segment)] = segment

            segments.append(padded)

            return segments, "segmented_and_padded"

        segments.append(segment)

        start += TARGET_SAMPLES

    return segments, "segmented"


# ============================================================
# MAIN PROCESSING
# ============================================================

total_files = 0
processed_files = 0
excluded_files = 0

padded_files = 0
segmented_files = 0
segmented_and_padded_files = 0

output_segments = 0

print("\n===================================================")
print("       INDIAN LANGUAGE AUDIO PREPROCESSING")
print("===================================================")

for data_type in TYPES:

    for language in LANGUAGES:

        input_folder = BASE / data_type / language
        output_folder = OUTPUT_BASE / data_type / language

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        if data_type == "real":
            files = sorted(
                input_folder.glob("*.flac")
            )
        else:
            files = sorted(
                input_folder.glob("*.wav")
            )

        print(
            f"\nProcessing: "
            f"{data_type.upper()} - "
            f"{language.upper()}"
        )

        print(
            f"Input files: {len(files)}"
        )

        for file in files:

            total_files += 1

            try:

                audio, sr = sf.read(
                    file,
                    dtype="float32"
                )

                # --------------------------------------------
                # Convert to mono
                # --------------------------------------------

                if audio.ndim > 1:
                    audio = np.mean(
                        audio,
                        axis=1
                    )

                # --------------------------------------------
                # Resample to 16 kHz
                # --------------------------------------------

                audio = resample_audio(
                    audio,
                    sr,
                    TARGET_SR
                )

                # --------------------------------------------
                # Check invalid values
                # --------------------------------------------

                if not np.isfinite(audio).all():

                    print(
                        f"Skipping invalid audio: "
                        f"{file.name}"
                    )

                    excluded_files += 1
                    continue

                # --------------------------------------------
                # Normalize amplitude
                # --------------------------------------------

                audio = normalize_audio(audio)

                # --------------------------------------------
                # Create fixed-length segments
                # --------------------------------------------

                segments, status = create_segments(
                    audio
                )

                if status == "excluded_short":

                    excluded_files += 1

                    print(
                        f"Excluded (<1 sec): "
                        f"{file.name}"
                    )

                    continue

                processed_files += 1

                if status == "padded":
                    padded_files += 1

                elif status == "segmented":
                    segmented_files += 1

                elif status == "segmented_and_padded":
                    segmented_and_padded_files += 1

                # --------------------------------------------
                # Save segments
                # --------------------------------------------

                stem = file.stem

                for index, segment in enumerate(
                    segments,
                    start=1
                ):

                    output_name = (
                        f"{stem}_seg{index:03d}.wav"
                    )

                    output_path = (
                        output_folder /
                        output_name
                    )

                    sf.write(
                        output_path,
                        segment,
                        TARGET_SR,
                        subtype="PCM_16"
                    )

                    output_segments += 1

            except Exception as e:

                print(
                    f"ERROR: "
                    f"{file.name} | {e}"
                )

                excluded_files += 1


# ============================================================
# FINAL REPORT
# ============================================================

print("\n===================================================")
print("              PREPROCESSING COMPLETE")
print("===================================================")

print(
    "Total input files             :",
    total_files
)

print(
    "Processed files               :",
    processed_files
)

print(
    "Excluded files (<1 sec/error) :",
    excluded_files
)

print(
    "Files padded (1–5 sec)        :",
    padded_files
)

print(
    "Files segmented (>5 sec)      :",
    segmented_files
)

print(
    "Segmented + final padded      :",
    segmented_and_padded_files
)

print(
    "Total output segments         :",
    output_segments
)

print("===================================================")