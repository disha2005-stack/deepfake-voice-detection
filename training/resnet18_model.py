
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_resnet18(num_classes=2, pretrained=True):
    """
    ResNet18 adapted for single-channel Mel-spectrograms.

    Labels:
        0 = REAL
        1 = FAKE
    """

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Change input convolution from 3 channels to 1 channel.
    old_conv = model.conv1

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    if pretrained:
        # Average RGB filters to initialize grayscale filters.
        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
            )

    model.conv1 = new_conv

    # Replace ImageNet's 1000-class output layer.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = create_resnet18(pretrained=False)

    sample = torch.randn(2, 1, 224, 336)
    output = model(sample)

    print("ResNet18 model test")
    print("Input shape :", sample.shape)
    print("Output shape:", output.shape)
    print("Expected output: [2, 2]")
