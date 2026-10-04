# To verify the preprocessing is done correctly and normalized
import os
import soundfile as sf
import numpy as np

BASE_DIR = r"D:\PBL 5\dataset\cleaned"

EXPECTED_SR = 16000
MIN_DURATION = 1.0
MAX_DURATION = 15.0

total = 0
passed = 0
failed = 0

problems = []

for label in ["real", "fake"]:

    folder = os.path.join(BASE_DIR, label)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    print(f"\nChecking {label.upper()}...")
    print(f"Files found: {len(files)}")

    for filename in files:

        filepath = os.path.join(folder, filename)
        total += 1

        try:
            audio, sr = sf.read(filepath)

            # Check sample rate
            if sr != EXPECTED_SR:
                problems.append(
                    f"{label}/{filename}: Sample rate = {sr}"
                )
                failed += 1
                continue

            # Check mono
            if audio.ndim != 1:
                problems.append(
                    f"{label}/{filename}: Not mono"
                )
                failed += 1
                continue

            # Check empty audio
            if len(audio) == 0:
                problems.append(
                    f"{label}/{filename}: Empty audio"
                )
                failed += 1
                continue

            # Check NaN / infinity
            if not np.isfinite(audio).all():
                problems.append(
                    f"{label}/{filename}: Contains NaN/Inf"
                )
                failed += 1
                continue

            # Duration
            duration = len(audio) / sr

            if duration < MIN_DURATION or duration > MAX_DURATION:
                problems.append(
                    f"{label}/{filename}: Duration = {duration:.2f}s"
                )
                failed += 1
                continue

            # Peak amplitude
            peak = np.max(np.abs(audio))

            # Normalized PCM16 audio should be close to 1.0
            if peak == 0:
                problems.append(
                    f"{label}/{filename}: Silent audio"
                )
                failed += 1
                continue

            if peak < 0.95 or peak > 1.001:
                problems.append(
                    f"{label}/{filename}: Peak amplitude = {peak:.5f}"
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
print("PREPROCESSING VERIFICATION COMPLETE")
print("=" * 60)

print(f"Total files checked : {total}")
print(f"Passed              : {passed}")
print(f"Failed              : {failed}")

if total > 0:
    percentage = (passed / total) * 100
    print(f"Pass percentage     : {percentage:.2f}%")

print("=" * 60)

if failed == 0:

    print("✅ ALL FILES PASSED PREPROCESSING VERIFICATION")
    print("✅ Sample rate = 16 kHz")
    print("✅ Audio is mono")
    print("✅ Duration is between 1 and 15 seconds")
    print("✅ No NaN/Inf values")
    print("✅ Audio amplitude is normalized")
    print("✅ All files are readable")

else:

    print("\n⚠️ PROBLEMS FOUND:")
    
    for problem in problems[:50]:
        print(problem)

    if len(problems) > 50:
        print(f"\n...and {len(problems) - 50} more problems.")
