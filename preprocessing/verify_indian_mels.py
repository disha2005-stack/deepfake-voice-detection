from pathlib import Path
from PIL import Image
import numpy as np

BASE = Path("features/indian/mel_spectrograms")

languages = ["kannada", "hindi", "tamil", "telugu"]
classes = ["real", "fake"]

expected_counts = {
    ("real", "kannada"): 873,
    ("fake", "kannada"): 1201,
    ("real", "hindi"): 853,
    ("fake", "hindi"): 804,
    ("real", "tamil"): 885,
    ("fake", "tamil"): 934,
    ("real", "telugu"): 784,
    ("fake", "telugu"): 1069,
}

total = 0
readable = 0
corrupt = 0
blank = 0

dimensions = set()

print("\n===================================================")
print("       INDIAN MEL-SPECTROGRAM VERIFICATION")
print("===================================================")

for data_class in classes:
    for language in languages:

        folder = BASE / data_class / language
        files = sorted(folder.glob("*.png"))

        expected = expected_counts[(data_class, language)]

        local_readable = 0
        local_corrupt = 0
        local_blank = 0

        print(
            f"\n{data_class.upper():5} | "
            f"{language.upper():7} | "
            f"Files: {len(files):5} | "
            f"Expected: {expected:5}"
        )

        if len(files) != expected:
            print("       WARNING: File count does not match expected count!")

        for file in files:

            total += 1

            try:
                with Image.open(file) as img:

                    # Check image integrity
                    img.verify()

                # Reopen after verify()
                with Image.open(file) as img:

                    image = np.array(img)

                    width, height = img.size
                    dimensions.add((width, height))

                    # Check for completely blank image
                    if image.size == 0 or np.std(image) == 0:
                        blank += 1
                        local_blank += 1

                    readable += 1
                    local_readable += 1

            except Exception as e:

                corrupt += 1
                local_corrupt += 1

                print(
                    f"       CORRUPT: {file.name} | {e}"
                )

        print(
            f"       Readable: {local_readable:5} | "
            f"Corrupt: {local_corrupt:3} | "
            f"Blank: {local_blank:3}"
        )


print("\n===================================================")
print("                 FINAL RESULTS")
print("===================================================")

print(f"TOTAL PNG FILES       : {total}")
print(f"READABLE PNG FILES    : {readable}")
print(f"CORRUPT PNG FILES     : {corrupt}")
print(f"BLANK PNG FILES       : {blank}")

print(f"IMAGE DIMENSIONS      : {sorted(dimensions)}")

print("===================================================")

if (
    total == 7403
    and readable == 7403
    and corrupt == 0
    and blank == 0
):
    print("RESULT: ALL MEL-SPECTROGRAMS PASSED VERIFICATION")
else:
    print("RESULT: VERIFICATION NEEDS ATTENTION")

print("===================================================\n")