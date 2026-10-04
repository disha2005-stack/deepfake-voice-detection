from pathlib import Path
import re

BASE = Path("dataset/cleaned")

languages = ["kannada", "hindi", "tamil", "telugu"]
classes = ["real", "fake"]

print("\n===================================================")
print("             SOURCE GROUP COUNTS")
print("===================================================")

for data_class in classes:

    for language in languages:

        folder = BASE / data_class / language
        files = list(folder.glob("*.wav"))

        source_groups = set()

        for file in files:

            # Remove _seg001, _seg002, etc.
            source_id = re.sub(r"_seg\d+$", "", file.stem)

            source_groups.add(source_id)

        print(
            f"{data_class.upper():5} | "
            f"{language.upper():7} | "
            f"Segments: {len(files):4} | "
            f"Source groups: {len(source_groups):4}"
        )

print("===================================================")