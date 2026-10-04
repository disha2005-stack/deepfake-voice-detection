import torch

from dataset_loader import train_dataset


print("\n===================================================")
print("       SPEC AUGMENTATION VERIFICATION")
print("===================================================")

image_original, label = train_dataset[0]

# Find the SpecAugment transform
spec_augment = train_dataset.transform.transforms[-1]

# Apply augmentation repeatedly
differences = []

for i in range(10):

    augmented = spec_augment(image_original.clone())

    difference = torch.abs(
        image_original - augmented
    ).sum().item()

    differences.append(difference)

    print(f"Test {i + 1:2}: Difference = {difference:.4f}")


print("\n===================================================")

changed = sum(d > 0 for d in differences)

print(f"Changed samples : {changed}/10")

if changed > 0:
    print("RESULT: SPEC AUGMENTATION IS WORKING")
else:
    print("RESULT: NO CHANGES DETECTED")

print("===================================================\n")