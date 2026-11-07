"""Central configuration for the whole project.

Every path and tunable number lives here so that the training scripts, the
command-line tool and the web app always agree with each other.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Folders
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
MRI_DATA_DIR = DATASETS_DIR / "mri"
SPEECH_DATA_DIR = DATASETS_DIR / "speech"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

MRI_MODEL_PATH = MODELS_DIR / "mri_resnet50.pt"
SPEECH_MODEL_PATH = MODELS_DIR / "speech_model.joblib"

# --------------------------------------------------------------------------
# MRI module
# --------------------------------------------------------------------------
# Classes are kept in order of severity; the index is used as a severity score.
MRI_CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
MRI_CLASS_LABELS = {
    "NonDemented": "Non Demented",
    "VeryMildDemented": "Very Mild Demented",
    "MildDemented": "Mild Demented",
    "ModerateDemented": "Moderate Demented",
}
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TEST_SPLIT = 0.20          # 80 : 20 train/test split, as in the presentation
RANDOM_SEED = 42

# Kaggle dataset used in the presentation (44,000 images, 4 classes)
KAGGLE_MRI_DATASET = "aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented"

# --------------------------------------------------------------------------
# Speech module
# --------------------------------------------------------------------------
SPEECH_CLASSES = ["Control", "Dementia"]
WHISPER_MODEL_SIZE = "base.en"     # tiny.en / base.en / small.en  (bigger = slower, more accurate)
SYNTHETIC_SPEECH_CSV = SPEECH_DATA_DIR / "SYNTHETIC_demo_transcripts.csv"

# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
# Used only when a trained model does not report its own test accuracy.
DEFAULT_MRI_WEIGHT = 0.6
DEFAULT_SPEECH_WEIGHT = 0.4
RISK_LOW_BELOW = 0.35      # fused score below this  -> Low
RISK_HIGH_ABOVE = 0.65     # fused score above this  -> High

DISCLAIMER = (
    "This is a research and educational prototype. It is not a medical device and "
    "its output is not a diagnosis. Only a qualified clinician can diagnose "
    "Alzheimer's disease, using clinical examination and validated tests."
)
