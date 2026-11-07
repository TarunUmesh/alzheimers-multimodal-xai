"""Grad-CAM for the ResNet50 MRI classifier, plus a region summary of the heat-map.

Grad-CAM (Selvaraju et al., 2017) answers "which parts of the image pushed the
model towards this class?". It takes the feature maps of the last
convolutional block, weights each map by how strongly the class score depends
on it (the gradient), and adds them up into a coarse heat-map.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.layer4
        self._activations = None
        self._hook = self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self._activations = output
        if output.requires_grad:
            output.retain_grad()

    def close(self) -> None:
        self._hook.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1, 3, H, W). Returns (cam HxW in [0,1], probabilities, class_idx)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)       # forces a graph even if weights are frozen
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            if class_idx is None:
                class_idx = int(probs.argmax())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()

        activations = self._activations[0]                    # (C, h, w)
        gradients = self._activations.grad[0]                 # (C, h, w)
        weights = gradients.mean(dim=(1, 2))                  # importance of each feature map
        cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, probs.detach().cpu().numpy(), class_idx


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend the heat-map over the MRI. Hot (red) = most influential."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heat = colormaps["jet"](cam)[..., :3].astype(np.float32)
    weight = (alpha * cam)[..., None]              # cold areas stay almost untouched
    blended = base * (1 - weight) + heat * weight
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


def brain_mask(image: Image.Image) -> np.ndarray:
    """Rough foreground mask: everything that is not the black background."""
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    mask = grey > max(12.0, 0.08 * float(grey.max()))
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    labelled, n = ndimage.label(mask)
    if n > 1:                                      # keep the largest blob only
        sizes = ndimage.sum(mask, labelled, range(1, n + 1))
        mask = labelled == (int(np.argmax(sizes)) + 1)
    return mask


def summarise_regions(image: Image.Image, cam: np.ndarray) -> dict:
    """Describe where the Grad-CAM heat sits, using coarse image zones.

    The zones are geometric approximations on a 2-D slice, not an anatomical
    segmentation, and are reported as such.

    For every zone we report
      heat_share - fraction of all heat that falls in the zone
      area_share - fraction of the brain area the zone covers
      focus      - heat_share / area_share  (>1 means the model looked there
                   more than its size alone would explain)
    """
    mask = brain_mask(image)
    total_heat = float(cam.sum()) + 1e-8
    inside_share = float(cam[mask].sum()) / total_heat
    result = {"inside_brain_share": inside_share, "zones": {}, "reliable": True, "peak": None}

    if mask.sum() < 0.05 * mask.size:
        result["reliable"] = False
        result["note"] = "Could not find a clear brain outline in the image."
        return result

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    height, width = (y1 - y0 + 1), (x1 - x0 + 1)

    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    # Central ellipse = roughly where the lateral ventricles sit on an axial slice.
    central = (((yy - cy) / (0.28 * height)) ** 2 + ((xx - cx) / (0.22 * width)) ** 2 <= 1) & mask
    # Outer ring = cortex (grey matter folds at the edge of the brain).
    ring_px = max(3, int(0.09 * min(height, width)))
    outer = mask & ~ndimage.binary_erosion(mask, iterations=ring_px)
    middle = mask & ~central & ~outer

    zones = {
        "central": central,
        "outer_cortex": outer,
        "deep_middle": middle,
        "image_left": mask & (xx < cx),
        "image_right": mask & (xx >= cx),
        "front_half": mask & (yy < cy),
        "back_half": mask & (yy >= cy),
    }
    brain_heat = float(cam[mask].sum()) + 1e-8
    brain_area = float(mask.sum())
    for name, zone in zones.items():
        heat_share = float(cam[zone].sum()) / brain_heat
        area_share = float(zone.sum()) / brain_area
        result["zones"][name] = {
            "heat_share": heat_share,
            "area_share": area_share,
            "focus": heat_share / area_share if area_share > 0 else 0.0,
        }

    py, px = np.unravel_index(int(np.argmax(np.where(mask, cam, -1))), cam.shape)
    result["peak"] = {
        "x": float((px - x0) / max(width, 1)),      # 0 = left edge of brain, 1 = right edge
        "y": float((py - y0) / max(height, 1)),     # 0 = top (front), 1 = bottom (back)
    }
    if inside_share < 0.6:
        result["reliable"] = False
        result["note"] = (
            f"Only {inside_share:.0%} of the heat-map lies on brain tissue; the rest is on the "
            "background, so this explanation should not be trusted."
        )
    return result
