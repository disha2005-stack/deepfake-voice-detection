import os
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Change this to the location of your trial_metadata.txt
METADATA_FILE = Path(
     r"D:\PBL 5\metadata\trial_metadata.txt"
)

# Change this to the folder containing the ASVspoof DF FLAC files
# Example:
# D:\PBL 5\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac
AUDIO_ROOT = Path(
   r"D:\PBL 5\ASVspoof2021_DF_eval_part00\ASVspoof2021_DF_eval\flac"
)

# Where our small test dataset will be created
OUTPUT_ROOT = Path(
     r"D:\PBL 5\dataset\raw"
)

# Number of samples per class
SAMPLES_PER_CLASS = 20


# ============================================================
# STEP 1: READ METADATA
# ============================================================

def read_metadata():

    real_files = []
    fake_files = []

    print("\nReading metadata...")
    print(f"Metadata file: {METADATA_FILE}")

    with open(METADATA_FILE, "r", encoding="utf-8", errors="ignore") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            columns = line.split()

            # We expect at least:
            # 0 = source ID
            # 1 = audio ID
            # 2 = codec
            # 3 = dataset
            # 4 = attack ID
            # 5 = label

            if len(columns) < 6:
                continue

            audio_id = columns[1]
            label = columns[5].lower()

            # REAL
            if label == "bonafide":

                if len(real_files) < SAMPLES_PER_CLASS:
                    real_files.append(audio_id)

            # FAKE
            elif label == "spoof":

                if len(fake_files) < SAMPLES_PER_CLASS:
                    fake_files.append(audio_id)

            # Stop reading once we have enough
            if (
                len(real_files) >= SAMPLES_PER_CLASS
                and len(fake_files) >= SAMPLES_PER_CLASS
            ):
                break

    print("\nMetadata selection complete.")

    print(f"REAL samples found : {len(real_files)}")
    print(f"FAKE samples found : {len(fake_files)}")

    return real_files, fake_files


# ============================================================
# STEP 2: FIND THE FLAC FILES
# ============================================================

def find_audio_files(required_ids):

    print("\nSearching for matching FLAC files...")
    print(f"Searching inside: {AUDIO_ROOT}")

    required_names = {
        audio_id.lower() + ".flac": audio_id
        for audio_id in required_ids
    }

    found_files = {}

    # Walk through folders recursively
    for root, dirs, files in os.walk(AUDIO_ROOT):

        for filename in files:

            filename_lower = filename.lower()

            if filename_lower in required_names:

                audio_id = required_names[filename_lower]

                full_path = Path(root) / filename

                found_files[audio_id] = full_path

                print(f"Found: {filename}")

                # Stop once all required files are found
                if len(found_files) == len(required_ids):
                    return found_files

    return found_files


# ============================================================
# STEP 3: COPY FILES INTO OUR DATASET
# ============================================================

def copy_files(file_ids, found_files, destination_folder):

    if destination_folder.exists() and not destination_folder.is_dir():
        raise RuntimeError(
            f"{destination_folder} exists but is not a folder. "
            "Please delete it and create a folder with that name."
        )

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    copied = 0

    for audio_id in file_ids:

        if audio_id not in found_files:

            print(f"WARNING: File not found: {audio_id}.flac")
            continue

        source = found_files[audio_id]

        destination = destination_folder / source.name

        shutil.copy2(source, destination)

        copied += 1

        print(f"Copied: {source.name}")

    return copied


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("ASVSPOOF TEST DATASET PREPARATION")
    print("=" * 60)

    # Check metadata
    if not METADATA_FILE.exists():

        print("\nERROR:")
        print("Metadata file was not found:")
        print(METADATA_FILE)

        return

    # Check audio directory
    if not AUDIO_ROOT.exists():

        print("\nERROR:")
        print("Audio directory was not found:")
        print(AUDIO_ROOT)

        return

    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    real_ids, fake_ids = read_metadata()

    required_ids = real_ids + fake_ids

    # --------------------------------------------------------
    # Find matching audio files
    # --------------------------------------------------------

    found_files = find_audio_files(required_ids)

    print("\n" + "=" * 60)
    print("FILE SEARCH RESULT")
    print("=" * 60)

    print(f"Required files : {len(required_ids)}")
    print(f"Found files    : {len(found_files)}")
    print(f"Missing files  : {len(required_ids) - len(found_files)}")

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    real_output = OUTPUT_ROOT / "real"
    fake_output = OUTPUT_ROOT / "fake"

    # --------------------------------------------------------
    # Copy REAL files
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("COPYING REAL AUDIO")
    print("=" * 60)

    real_copied = copy_files(
        real_ids,
        found_files,
        real_output
    )

    # --------------------------------------------------------
    # Copy FAKE files
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("COPYING FAKE AUDIO")
    print("=" * 60)

    fake_copied = copy_files(
        fake_ids,
        found_files,
        fake_output
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"REAL files copied : {real_copied}")
    print(f"FAKE files copied : {fake_copied}")

    print("\nDataset location:")
    print(OUTPUT_ROOT)

    print("\nExpected structure:")
    print(r"D:\PBL 5\dataset\raw\real")
    print(r"D:\PBL 5\dataset\raw\fake")


if __name__ == "__main__":
    main()