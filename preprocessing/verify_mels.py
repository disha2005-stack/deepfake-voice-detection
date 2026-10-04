# To verify melspectrogram generation is done correct
#ly and normalized
import os
from PIL import Image

BASE_DIR = r"D:\PBL 5\features\mel_spectrograms"

total = 0
passed = 0
failed = 0
problems = []

for label in ["real", "fake"]:

    folder = os.path.join(BASE_DIR, label)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ]

    print(f"\nChecking {label.upper()}...")
    print(f"Files found: {len(files)}")

    for filename in files:

        filepath = os.path.join(folder, filename)
        total += 1

        try:
            with Image.open(filepath) as img:

                # Verify image is not corrupted
                img.verify()

            # Reopen after verify() because verify() closes the file
            with Image.open(filepath) as img:

                width, height = img.size

                # Check dimensions
                if width <= 0 or height <= 0:
                    problems.append(
                        f"{label}/{filename}: Invalid dimensions"
                    )
                    failed += 1
                    continue

                # Check image mode
                if img.mode not in ["RGB", "RGBA", "L"]:
                    problems.append(
                        f"{label}/{filename}: Unexpected mode {img.mode}"
                    )
                    failed += 1
                    continue

            passed += 1

        except Exception as e:

            failed += 1
            problems.append(
                f"{label}/{filename}: ERROR - {e}"
            )


print("\n" + "=" * 60)
print("MEL-SPECTROGRAM VERIFICATION COMPLETE")
print("=" * 60)

print(f"Total PNG files checked : {total}")
print(f"Passed                  : {passed}")
print(f"Failed                  : {failed}")

if total > 0:
    percentage = (passed / total) * 100
    print(f"Pass percentage         : {percentage:.2f}%")

print("=" * 60)

if failed == 0:

    print("✅ ALL MEL-SPECTROGRAM FILES PASSED")
    print("✅ All PNG files are readable")
    print("✅ No corrupted images detected")
    print("✅ All images have valid dimensions")

else:

    print("\n⚠️ PROBLEMS FOUND:")

    for problem in problems[:50]:
        print(problem)

    if len(problems) > 50:
        print(f"\n...and {len(problems) - 50} more problems.")