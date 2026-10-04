import soundfile as sf
import numpy as np
from pathlib import Path

folder = Path("dataset/raw/fake/telugu")

files = list(folder.glob("*.wav"))

valid = 0
invalid = 0
sample_rates = set()
channels = set()
durations = []

for file in files:
    try:
        audio, sr = sf.read(file)

        if audio.ndim == 1:
            ch = 1
        else:
            ch = audio.shape[1]

        duration = len(audio) / sr

        if (
            sr == 16000
            and ch == 1
            and 1 <= duration <= 15
            and np.isfinite(audio).all()
        ):
            valid += 1
        else:
            invalid += 1

        sample_rates.add(sr)
        channels.add(ch)
        durations.append(duration)

    except Exception as e:
        invalid += 1
        print("ERROR:", file.name, "|", e)

print("\n========== TELUGU FAKE AUDIO VERIFICATION ==========")
print("Total files       :", len(files))
print("Valid files       :", valid)
print("Invalid files     :", invalid)
print("Sample rates      :", sorted(sample_rates))
print("Channels          :", sorted(channels))
print("Minimum duration  :", round(min(durations), 3), "sec")
print("Maximum duration  :", round(max(durations), 3), "sec")
print("Average duration  :", round(sum(durations) / len(durations), 3), "sec")