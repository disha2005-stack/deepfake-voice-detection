from datasets import load_dataset

print("Loading Kannada dataset...")

dataset = load_dataset(
    "ai4bharat/IndicVoices",
    "kannada",
    split="train",
    streaming=True
)

# Disable automatic audio decoding
dataset = dataset.cast_column(
    "audio_filepath",
    dataset.features["audio_filepath"].__class__(
        decode=False
    )
)

print("\nReading one record...")

sample = next(iter(dataset))

print("\n===== AUDIO INFORMATION =====")
print(sample["audio_filepath"])

print("\n===== OTHER INFORMATION =====")
print("Language:", sample["lang"])
print("Duration:", sample["duration"])
print("Speaker ID:", sample["speaker_id"])
print("Gender:", sample["gender"])
print("Age group:", sample["age_group"])
print("State:", sample["state"])
print("District:", sample["district"])
print("Scenario:", sample["scenario"])
print("Task:", sample["task_name"])
print("Text:", sample["text"])