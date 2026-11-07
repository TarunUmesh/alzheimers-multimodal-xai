# Datasets

Two datasets are used, one per modality. They are separate collections: no
public dataset has an MRI scan and a speech recording from the same patients.

## 1. MRI — Alzheimer's Disease Multiclass Images Dataset (Kaggle)

| | |
|---|---|
| Link | https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented |
| Size | 44,000 brain MRI slices |
| Classes | NonDemented 12,800 · VeryMildDemented 11,200 · MildDemented 10,000 · ModerateDemented 10,000 |
| Split used | 80 % train / 20 % test (8,800 test images) |

**Automatic download**

```
python scripts/download_datasets.py
```

**Manual download** (if the automatic one fails)

1. Open the link above and sign in to Kaggle (free).
2. Click **Download** and unzip the file.
3. Copy the four class folders into `datasets/mri/`.

The loader accepts any folder nesting and the common spellings of the class
names, so other 4-class Alzheimer MRI datasets from Kaggle also work
(`python scripts/download_datasets.py --slug owner/dataset-name`).

Note for the viva: this dataset is an *augmented* version of a much smaller
set of original scans. Augmented copies of one scan can land on both sides of
a random train/test split, which makes test accuracy look better than it
would be on scans from new patients.

## 2. Speech — DementiaBank Pitt Corpus

| | |
|---|---|
| Link | https://talkbank.org/dementia/access/English/Pitt.html |
| Content | Transcripts and audio of the "Cookie Theft" picture description task |
| Classes | Control · Dementia |
| Citation | Becker, J. T., Boller, F., Lopez, O. L., Saxton, J., & McGonigle, K. L. (1994). The natural history of Alzheimer's disease: description of study cohort and accuracy of diagnosis. *Archives of Neurology, 51*(6), 585–594. |

DementiaBank contains recordings of real patients, so it **cannot be
downloaded freely or redistributed**. Access is given on request (through
talkbank.org, or by email to talkbank@cmu.edu) — for students, normally
through a faculty member.

When you have it, put **either** of these in `datasets/speech/`:

- the original folders `Control/cookie/*.cha` and `Dementia/cookie/*.cha`, or
- a CSV file with two columns, `text` and `label` (`Control` / `Dementia`).

Then run `python scripts/train_speech.py`. Real data in this folder is used
automatically in preference to the demo file.

### `speech/SYNTHETIC_demo_transcripts.csv` — demo file, not patient data

500 computer-generated picture descriptions (250 per class) produced by
`scripts/make_synthetic_speech_data.py`. It exists so the speech pipeline can
be run and demonstrated before DementiaBank access arrives.

- It is **not** DementiaBank and contains no real speech.
- Accuracy measured on it is **not** a research result and must not be
  reported as one.
- The app and the report print a warning whenever the speech model in use was
  trained on it.
