import soundfile as sf
import numpy as np
from pathlib import Path

BASE = Path("dataset/raw")

languages = ["kannada", "hindi", "tamil", "telugu"]
types = ["real", "fake"]

total = 0
readable = 0
corrupt = 0
nan_inf = 0

print("\n========== INDIAN DATASET QUALITY AUDIT ==========\n")

for data_type in types:
    for language in languages:

        folder = BASE / data_type / language

        if data_type == "real":
            files = list(folder.glob("*.flac"))
        else:
            files = list(folder.glob("*.wav"))

        print(f"{data_type.upper():5} | {language.upper():7} | Files: {len(files)}")

        sample_rates = set()
        channels = set()
        durations = []

        local_corrupt = 0
        local_nan_inf = 0

        for file in files:
            total += 1

            try:
                audio, sr = sf.read(file)

                readable += 1

                if audio.ndim == 1:
                    ch = 1
                else:
                    ch = audio.shape[1]

                duration = len(audio) / sr

                sample_rates.add(sr)
                channels.add(ch)
                durations.append(duration)

                if not np.isfinite(audio).all():
                    nan_inf += 1
                    local_nan_inf += 1

            except Exception as e:
                corrupt += 1
                local_corrupt += 1
                print(f"  ERROR: {file.name} | {e}")

        if durations:
            print(f"       Sample rates : {sorted(sample_rates)}")
            print(f"       Channels     : {sorted(channels)}")
            print(f"       Min duration : {min(durations):.2f} sec")
            print(f"       Max duration : {max(durations):.2f} sec")
            print(f"       Avg duration : {sum(durations)/len(durations):.2f} sec")

        print(f"       Corrupt      : {local_corrupt}")
        print(f"       NaN/Inf      : {local_nan_inf}")
        print()

print("===================================================")
print("TOTAL FILES        :", total)
print("READABLE FILES     :", readable)
print("CORRUPT FILES      :", corrupt)
print("FILES WITH NaN/Inf :", nan_inf)
print("===================================================")