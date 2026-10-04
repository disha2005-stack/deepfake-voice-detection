import torch


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Class weights calculated from the training set
CLASS_WEIGHTS = torch.tensor(
    [1.083091, 0.928750],
    dtype=torch.float32
)


# Loss function
CRITERION = torch.nn.CrossEntropyLoss(
    weight=CLASS_WEIGHTS
)


# Optimizer settings
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 30


# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("\n===================================================")
print("             TRAINING CONFIGURATION")
print("===================================================")

print(f"\nDevice          : {DEVICE}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Learning rate   : {LEARNING_RATE}")
print(f"Weight decay    : {WEIGHT_DECAY}")
print(f"Number of epochs: {NUM_EPOCHS}")

print("\nClass weights:")
print(f"REAL            : {CLASS_WEIGHTS[0].item():.6f}")
print(f"FAKE            : {CLASS_WEIGHTS[1].item():.6f}")

print("\nLoss function:")
print(CRITERION)

print("\n===================================================")
print("RESULT: TRAINING CONFIGURATION LOADED")
print("===================================================\n")