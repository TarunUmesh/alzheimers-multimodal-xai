"""ResNet50 classifier for the four Alzheimer's stages."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

import config


def build_model(pretrained: bool = True, num_classes: int = len(config.MRI_CLASSES)) -> nn.Module:
    """ResNet50 backbone + a small fully connected classification head.

    `pretrained=True` downloads the ImageNet weights once (about 100 MB) and
    caches them; this is what makes transfer learning work.
    """
    weights = None
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    try:
        model = models.resnet50(weights=weights)
    except Exception as err:  # no internet -> fall back so the code still runs
        print(f"[warning] Could not download ImageNet weights ({err}).")
        print("[warning] Training from random weights - accuracy will be much lower.")
        model = models.resnet50(weights=None)

    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.LayerNorm(in_features),        # keeps the head stable whatever the feature scale
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def save_checkpoint(model: nn.Module, meta: dict, path: Path = config.MRI_MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": config.MRI_CLASSES, "meta": meta}, path)


def load_model(path: Path = config.MRI_MODEL_PATH, device: str | torch.device = "cpu"):
    """Load a trained checkpoint. Returns (model, meta)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained MRI model at {path}. Train one first with:  python scripts/train_mri.py"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt.get("meta", {})


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
