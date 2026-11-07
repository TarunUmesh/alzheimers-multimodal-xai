"""Train the ResNet50 MRI classifier.

Two modes:

fast      The ResNet50 backbone is frozen. Every image is passed through it once,
          the 2048-number feature vectors are cached on disk, and only the small
          fully connected head is trained. Practical on a laptop CPU.

finetune  The last ResNet block (layer4) is trained together with the head, with
          light augmentation. More accurate, but needs a GPU to be practical.

Split: 80 % train / 20 % test (as in the presentation). 10 % of the training part
is held out as a validation set for choosing the best epoch, so the test set is
never used for any decision.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from src.mri.data import MRIDataset, find_images, limit_per_class
from src.mri.model import build_model, pick_device, save_checkpoint


def _split(paths, labels, seed):
    train_p, test_p, train_y, test_y = train_test_split(
        paths, labels, test_size=config.TEST_SPLIT, stratify=labels, random_state=seed)
    train_p, val_p, train_y, val_y = train_test_split(
        train_p, train_y, test_size=0.10, stratify=train_y, random_state=seed)
    return (train_p, train_y), (val_p, val_y), (test_p, test_y)


def _loader(paths, labels, batch_size, workers, augment=False, shuffle=False):
    return DataLoader(MRIDataset(paths, labels, augment=augment), batch_size=batch_size,
                      shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


# --------------------------------------------------------------------------
# fast mode
# --------------------------------------------------------------------------
@torch.no_grad()
def _extract_features(model, paths, labels, device, batch_size, workers, cache_dir: Path, tag: str):
    """Backbone features for every image, cached so a re-run takes seconds."""
    # The cache key covers the file list AND the backbone weights, so a different
    # dataset or different starting weights never reuses stale features.
    fingerprint = f"{float(model.conv1.weight.sum()):.6f}|" + "|".join(str(p) for p in paths)
    digest = hashlib.md5(fingerprint.encode()).hexdigest()[:12]
    cache_file = cache_dir / f"features_{tag}_{digest}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        print(f"  {tag}: loaded cached features ({len(data['y'])} images)")
        return torch.from_numpy(data["x"]), torch.from_numpy(data["y"])

    head, model.fc = model.fc, nn.Identity()
    model.eval()
    feats = []
    for x, _ in tqdm(_loader(paths, labels, batch_size, workers), desc=f"  features [{tag}]"):
        feats.append(model(x.to(device)).cpu())
    model.fc = head
    x_all = torch.cat(feats)
    y_all = torch.tensor(labels, dtype=torch.long)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, x=x_all.numpy(), y=y_all.numpy())
    return x_all, y_all


def _train_fast(model, splits, device, epochs, batch_size, workers, lr, cache_dir):
    (train_p, train_y), (val_p, val_y), (test_p, test_y) = splits
    print("Step 1/2: passing images through the frozen ResNet50 backbone")
    x_tr, y_tr = _extract_features(model, train_p, train_y, device, batch_size, workers, cache_dir, "train")
    x_va, y_va = _extract_features(model, val_p, val_y, device, batch_size, workers, cache_dir, "val")
    x_te, y_te = _extract_features(model, test_p, test_y, device, batch_size, workers, cache_dir, "test")

    print("Step 2/2: training the fully connected head")
    head = model.fc.to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=256, shuffle=True)

    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        head.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(xb)
            seen += len(xb)
        scheduler.step()
        head.eval()
        with torch.no_grad():
            val_acc = (head(x_va.to(device)).argmax(1).cpu() == y_va).float().mean().item()
        history.append({"epoch": epoch, "train_loss": running / seen, "val_accuracy": val_acc})
        if val_acc >= best_acc:          # on a tie keep the later, better-fitted epoch
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs}  loss {running / seen:.4f}  val acc {val_acc:.4f}")

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        preds = head(x_te.to(device)).argmax(1).cpu().numpy()
    return preds, y_te.numpy(), history


# --------------------------------------------------------------------------
# finetune mode
# --------------------------------------------------------------------------
def _set_train_mode(model):
    """Only layer4 + head learn; frozen layers stay in eval mode (keeps BatchNorm stats)."""
    model.eval()
    model.layer4.train()
    model.fc.train()


@torch.no_grad()
def _predict_loader(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        targets.append(y)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def _train_finetune(model, splits, device, epochs, batch_size, workers, lr):
    (train_p, train_y), (val_p, val_y), (test_p, test_y) = splits
    for p in model.parameters():
        p.requires_grad = False
    for p in list(model.layer4.parameters()) + list(model.fc.parameters()):
        p.requires_grad = True

    train_loader = _loader(train_p, train_y, batch_size, workers, augment=True, shuffle=True)
    val_loader = _loader(val_p, val_y, batch_size, workers)
    test_loader = _loader(test_p, test_y, batch_size, workers)

    optimiser = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": lr / 10},
        {"params": model.fc.parameters(), "lr": lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state, history = -1.0, None, []
    for epoch in range(1, epochs + 1):
        _set_train_mode(model)
        running, seen = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"  epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
            running += loss.item() * len(x)
            seen += len(x)
        scheduler.step()
        preds, targets = _predict_loader(model, val_loader, device)
        val_acc = float((preds == targets).mean())
        history.append({"epoch": epoch, "train_loss": running / seen, "val_accuracy": val_acc})
        print(f"  epoch {epoch}/{epochs}  loss {running / seen:.4f}  val acc {val_acc:.4f}")
        if val_acc >= best_acc:          # on a tie keep the later, better-fitted epoch
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    preds, targets = _predict_loader(model, test_loader, device)
    return preds, targets, history


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _save_confusion_matrix(cm, classes, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Alzheimer Classification Confusion Matrix")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(data_dir: Path = config.MRI_DATA_DIR, mode: str = "auto", epochs: int | None = None,
          batch_size: int = 32, workers: int = 0, lr: float = 1e-3,
          max_per_class: int | None = None, seed: int = config.RANDOM_SEED,
          model_path: Path = config.MRI_MODEL_PATH, output_dir: Path = config.OUTPUTS_DIR,
          pretrained: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()
    if mode == "auto":
        mode = "finetune" if device.type == "cuda" else "fast"
    if epochs is None:
        epochs = 30 if mode == "fast" else 8

    paths, labels = find_images(data_dir)
    if not paths:
        raise SystemExit(
            f"No MRI images found under {data_dir}.\n"
            "Run  python scripts/download_datasets.py  or see datasets/README.md."
        )
    paths, labels = limit_per_class(paths, labels, max_per_class, seed)
    counts = {c: labels.count(i) for i, c in enumerate(config.MRI_CLASSES)}
    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        raise SystemExit(f"No images found for class(es): {missing}. Check the folder names in {data_dir}.")
    print(f"Device: {device}   mode: {mode}   epochs: {epochs}")
    print(f"Images: {len(paths)}  {counts}")

    splits = _split(paths, labels, seed)
    print(f"Split: train {len(splits[0][0])}  val {len(splits[1][0])}  test {len(splits[2][0])}")

    model = build_model(pretrained=pretrained).to(device)
    started = time.time()
    if mode == "fast":
        preds, targets, history = _train_fast(model, splits, device, epochs, batch_size, workers, lr,
                                              Path(output_dir) / "cache")
    else:
        preds, targets, history = _train_finetune(model, splits, device, epochs, batch_size, workers, lr)
    minutes = (time.time() - started) / 60

    accuracy = float(accuracy_score(targets, preds))
    report_text = classification_report(targets, preds, target_names=config.MRI_CLASSES, digits=4, zero_division=0)
    report = classification_report(targets, preds, target_names=config.MRI_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(len(config.MRI_CLASSES))))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrix(cm, config.MRI_CLASSES, output_dir / "mri_confusion_matrix.png")
    (output_dir / "mri_classification_report.txt").write_text(report_text, encoding="utf-8")

    meta = {
        "test_accuracy": accuracy,
        "mode": mode,
        "epochs": epochs,
        "n_images": len(paths),
        "class_counts": counts,
        "n_test": int(len(targets)),
        "pretrained_backbone": bool(pretrained),
        "trained_on": time.strftime("%Y-%m-%d %H:%M"),
        "training_minutes": round(minutes, 1),
    }
    (output_dir / "mri_metrics.json").write_text(json.dumps(
        {**meta, "report": report, "confusion_matrix": cm.tolist(), "history": history}, indent=2), encoding="utf-8")
    save_checkpoint(model.cpu(), meta, model_path)

    print("\n" + report_text)
    print(f"Test accuracy: {accuracy:.4f}   (training took {minutes:.1f} min)")
    print(f"Model saved to   {model_path}")
    print(f"Reports saved to {output_dir}")
    return meta
