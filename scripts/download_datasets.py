"""Download the MRI dataset into datasets/mri and explain how to get the speech dataset.

Usage
  python scripts/download_datasets.py
  python scripts/download_datasets.py --slug owner/another-kaggle-dataset
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.mri.data import find_images  # noqa: E402

SPEECH_HELP = f"""
SPEECH DATASET (DementiaBank Pitt Corpus)
-----------------------------------------
DementiaBank cannot be downloaded automatically: it holds recordings of real
patients and access is given to researchers on request.
  1. Read https://talkbank.org/dementia/access/English/Pitt.html
  2. Ask your professor / guide to request access (email talkbank@cmu.edu or
     join DementiaBank through talkbank.org). Students normally go through a
     faculty member.
  3. When you have it, put EITHER
        the Pitt folder with  Control/cookie/*.cha  and  Dementia/cookie/*.cha
     OR a CSV file with the columns  text,label   (label = Control or Dementia)
     inside  {config.SPEECH_DATA_DIR}
  4. Run  python scripts/train_speech.py

Until then, {config.SYNTHETIC_SPEECH_CSV.name} (computer-generated,
NOT patient data) lets the whole pipeline run.
"""


def report_counts() -> int:
    paths, labels = find_images(config.MRI_DATA_DIR)
    print(f"\nMRI images found in {config.MRI_DATA_DIR}: {len(paths)}")
    for i, cls in enumerate(config.MRI_CLASSES):
        print(f"  {cls:18s} {labels.count(i)}")
    return len(paths)


def download_mri(slug: str) -> None:
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("Run  pip install kagglehub  first (it is in requirements.txt).")
    print(f"Downloading Kaggle dataset '{slug}' (44,000 images - this is a large download, please wait)...")
    try:
        cache_path = Path(kagglehub.dataset_download(slug))
    except Exception as err:
        raise SystemExit(
            f"\nAutomatic download failed: {err}\n\n"
            f"Download it by hand instead:\n"
            f"  1. Open https://www.kaggle.com/datasets/{slug}  and sign in (free)\n"
            f"  2. Click Download, then unzip\n"
            f"  3. Copy the four class folders into  {config.MRI_DATA_DIR}\n"
        )
    print(f"Downloaded to {cache_path}\nCopying into {config.MRI_DATA_DIR} ...")
    shutil.copytree(cache_path, config.MRI_DATA_DIR, dirs_exist_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=config.KAGGLE_MRI_DATASET, help="Kaggle dataset as owner/name")
    ap.add_argument("--force", action="store_true", help="download even if images are already present")
    args = ap.parse_args()

    existing, _ = find_images(config.MRI_DATA_DIR)
    if existing and not args.force:
        print("MRI images are already present - skipping download (use --force to download again).")
    else:
        download_mri(args.slug)
    if report_counts() == 0:
        print("No class folders were recognised. Expected folder names such as "
              "NonDemented / VeryMildDemented / MildDemented / ModerateDemented.")
    print(SPEECH_HELP)
