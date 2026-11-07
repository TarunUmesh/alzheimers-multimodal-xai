"""Run the trained MRI model on one image and explain the prediction."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

import config
from src.mri.data import image_to_tensor, preprocess_pil
from src.mri.gradcam import GradCAM, overlay_heatmap, summarise_regions
from src.mri.model import load_model


class MRIPredictor:
    def __init__(self, model_path: Path = config.MRI_MODEL_PATH, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.meta = load_model(model_path, self.device)
        self.cam = GradCAM(self.model)

    def predict(self, image: Image.Image | str | Path) -> dict:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        shown = preprocess_pil(image)                      # what the model actually sees
        x = image_to_tensor(image).unsqueeze(0).to(self.device)

        cam, probs, idx = self.cam(x)
        classes = config.MRI_CLASSES
        return {
            "predicted_class": classes[idx],
            "predicted_label": config.MRI_CLASS_LABELS[classes[idx]],
            "confidence": float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(classes, probs)},
            "p_impaired": float(1.0 - probs[classes.index("NonDemented")]),
            "preprocessed_image": shown,
            "gradcam": cam,
            "gradcam_overlay": overlay_heatmap(shown, cam),
            "regions": summarise_regions(shown, cam),
            "model_meta": self.meta,
        }
