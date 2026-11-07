"""Train the speech model.   Usage:  python scripts/train_speech.py  [--help]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.speech.train import train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the Control-vs-Dementia transcript classifier.")
    ap.add_argument("--data", type=Path, default=None,
                    help="CSV/TSV with text+label columns, or a folder of DementiaBank .cha files. "
                         "Default: real data in datasets/speech if present, otherwise the synthetic demo file.")
    ap.add_argument("--model", choices=["auto", "rf", "lr"], default="auto",
                    help="auto = keep whichever of Random Forest / Logistic Regression cross-validates better")
    args = ap.parse_args()
    train(data_path=args.data, model=args.model)
