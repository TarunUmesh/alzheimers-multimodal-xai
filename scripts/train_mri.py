"""Train the MRI model.   Usage:  python scripts/train_mri.py  [--help]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.mri.train import train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the ResNet50 Alzheimer MRI classifier.")
    ap.add_argument("--data", type=Path, default=config.MRI_DATA_DIR, help="folder with the class sub-folders")
    ap.add_argument("--mode", choices=["auto", "fast", "finetune"], default="auto",
                    help="fast = frozen backbone (CPU friendly); finetune = train layer4 too (GPU). "
                         "auto picks finetune when a GPU is found.")
    ap.add_argument("--epochs", type=int, default=None, help="default: 30 (fast) or 8 (finetune)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0, help="data-loading processes (0 is safest on Windows)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="use only this many images per class - handy for a quick trial run")
    args = ap.parse_args()
    train(data_dir=args.data, mode=args.mode, epochs=args.epochs, batch_size=args.batch_size,
          workers=args.workers, lr=args.lr, max_per_class=args.max_per_class)
