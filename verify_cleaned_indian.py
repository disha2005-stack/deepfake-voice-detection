import soundfile as sf
import numpy as np
from pathlib import Path

BASE = Path("dataset/cleaned")

languages = ["kannada", "hindi", "tamil", "telugu"]
classes = ["real", "fake"]

total = 0
valid = 0
invalid = 0

sample_rates = set()
channels = set()
durations = set()
subtypes = set()

print("\n===================================================")
print("       CLEANED INDIAN DATASET VERIFICATION")
print("===================================================")

for data_class in classes:

    for language in languages:

        folder = BASE / data_class / language
        files = sorted(folder.glob("*.wav"))

        class_valid = 0
        class_invalid = 0

        for file in files:

            total += 1

            try:
                info = sf.info(file)
                audio, sr = sf.read(
                    file,
                    dtype="float32"
                )

                sample_rates.add(sr)

                if audio.ndim == 1:
                    ch = 1
                else:
                    ch = audio.shape[1]

                channels.add(ch)

                duration = len(audio) / sr
                durations.add(round(duration, 6))

                subtypes.add(info.subtype)

                if (
                    sr == 16000
                    and ch == 1
                    and abs(duration - 5.0) < 0.001
                    and info.subtype == "PCM_16"
                    and np.isfinite(audio).all()
                ):
                    valid += 1
                    class_valid += 1
                else:
                    invalid += 1
                    class_invalid += 1

            except Exception as e:

                invalid += 1
                class_invalid += 1

                print(
                    f"ERROR: {file} | {e}"
                )

        print(
            f"{data_class.upper():5} | "
            f"{language.upper():7} | "
            f"Files: {len(files):5} | "
            f"Valid: {class_valid:5} | "
            f"Invalid: {class_invalid:5}"
        )

print("\n===================================================")
print("TOTAL FILES       :", total)
print("VALID FILES       :", valid)
print("INVALID FILES     :", invalid)
print("SAMPLE RATES      :", sorted(sample_rates))
print("CHANNELS          :", sorted(channels))
print("DURATIONS         :", sorted(durations))
print("AUDIO SUBTYPES    :", sorted(subtypes))
print("===================================================")