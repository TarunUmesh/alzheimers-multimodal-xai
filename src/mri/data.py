"""MRI dataset discovery, preprocessing and loading.

The loader is deliberately forgiving about folder names because the public
Alzheimer MRI datasets on Kaggle all name their class folders slightly
differently ("NonDemented", "Non Demented", "Non_Demented", ...). Any folder
whose name maps onto one of the four classes is picked up, at any depth.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def folder_to_class(name: str) -> str | None:
    """Map a folder name onto one of the four canonical class names."""
    key = re.sub(r"[^a-z]", "", name.lower())
    if "dement" not in key and "impair" not in key:
        return None
    if "non" in key or key.startswith("no"):
        return "NonDemented"
    if "verymild" in key:
        return "VeryMildDemented"
    if "moderate" in key:
        return "ModerateDemented"
    if "mild" in key:
        return "MildDemented"
    return None


def find_images(root: Path | str = config.MRI_DATA_DIR) -> tuple[list[Path], list[int]]:
    """Return (paths, labels) for every image under `root` that sits in a class folder."""
    root = Path(root)
    paths: list[Path] = []
    labels: list[int] = []
    if not root.exists():
        return paths, labels
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        # Nearest ancestor folder that looks like a class name decides the label.
        for parent in path.relative_to(root).parents:
            cls = folder_to_class(parent.name) if parent.name else None
            if cls:
                paths.append(path)
                labels.append(config.MRI_CLASSES.index(cls))
                break
    return paths, labels


def limit_per_class(paths, labels, max_per_class: int | None, seed: int = config.RANDOM_SEED):
    """Optionally keep only `max_per_class` random images per class (for quick runs)."""
    if not max_per_class:
        return paths, labels
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    keep: list[int] = []
    for c in sorted(set(labels)):
        idx = np.flatnonzero(labels_arr == c)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.extend(sorted(idx.tolist()))
    return [paths[i] for i in keep], [labels[i] for i in keep]


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def preprocess_pil(image: Image.Image) -> Image.Image:
    """Resize to 224x224, light denoise, 3-channel. Returns the image the model 'sees'."""
    image = image.convert("L")                                  # MRI slices are greyscale
    image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
    image = image.filter(ImageFilter.MedianFilter(size=3))      # removes speckle noise
    return image.convert("RGB")                                 # ResNet50 expects 3 channels


_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

_augment = transforms.Compose([
    transforms.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Full preprocessing for one image -> normalised tensor of shape (3, 224, 224)."""
    return _to_tensor(preprocess_pil(image))


class MRIDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], augment: bool = False):
        self.paths = paths
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as img:
            img = preprocess_pil(img)
        if self.augment:
            img = _augment(img)
        return _to_tensor(img), self.labels[i]
