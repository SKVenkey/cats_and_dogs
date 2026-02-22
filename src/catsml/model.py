"""CNN model definitions for Cats vs Dogs binary classification."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class SimpleCNN(nn.Module):
    """Lightweight custom CNN baseline."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TransferCNN(nn.Module):
    """ResNet-18 with fine-tuned head (recommended for best accuracy)."""

    def __init__(self, num_classes: int = 2, freeze_base: bool = False):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_base:
            for param in base.parameters():
                param.requires_grad = False
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_model(model_type: str = "simple_cnn", num_classes: int = 2) -> nn.Module:
    """Factory to get a model by name."""
    if model_type == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    elif model_type == "transfer_resnet18":
        return TransferCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_model(model_path: str, model_type: str = "simple_cnn", device: Optional[str] = None) -> nn.Module:
    """Load a saved model checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
