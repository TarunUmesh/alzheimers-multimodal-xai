# Multimodal Explainable AI for Early Detection of Alzheimer's Disease

Brain MRI analysis and speech analysis combined into one explained result.

| Part | What it does | How |
|---|---|---|
| MRI module | Classifies an MRI slice into 4 stages: Non / Very Mild / Mild / Moderate Demented | ResNet50 (transfer learning) |
| MRI explanation | Shows which part of the scan drove the decision | Grad-CAM heat-map + a written summary of where the heat lies |
| Speech-to-text | Turns a recording (microphone or file) into text, with word timings | Whisper, running locally (`faster-whisper`) |
| Speech module | Classifies the transcript as Control or Dementia | TF-IDF + 8 linguistic markers → Random Forest / Logistic Regression |
| Speech explanation | Shows which words and which speech habits drove the decision | Word-removal test + comparison of each marker with healthy speakers |
| Fusion | One combined risk score from both results | Accuracy-weighted late fusion |
| Reasons | Plain-language list of why the system said what it said | Rule-based, every sentence tied to a measured value |
| Guidance | Things to avoid, things to do, next steps — by stage | Fixed knowledge base from WHO / Lancet Commission / Alzheimer's Association |
| Report | Everything above as one downloadable HTML page | |

> **This is a research and educational prototype, not a medical device. Its output is not a diagnosis.**

---

## Run it in VS Code (Windows)

**Before you start:** install [Python](https://www.python.org/downloads/) 3.10 or newer (tick *Add python.exe to PATH*)
and the **Python** extension in VS Code.

1. Unzip the project and open the folder in VS Code (*File → Open Folder*).
2. Open a terminal (*Terminal → New Terminal*) and set up the environment — once only:

   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   (Or double-click `setup_windows.bat`.) If PowerShell refuses to run `activate`, switch the terminal to
   *Command Prompt*, or run `Set-ExecutionPolicy -Scope Process Bypass` first.
   The install is large (PyTorch) and takes several minutes.

3. Check that everything works — about a minute, no dataset needed:

   ```
   python tests/smoke_test.py
   ```

4. Get the MRI dataset (44,000 images - a large download):

   ```
   python scripts/download_datasets.py
   ```

5. Train the two models:

   ```
   python scripts/train_mri.py
   python scripts/train_speech.py
   ```

6. Start the app — it opens in your browser:

   ```
   streamlit run app.py
   ```

Steps 3–6 are also available from VS Code's **Run and Debug** panel (the ▶ icon on the left) as numbered entries.

### What to expect from training

**MRI.** On a computer without a graphics card the script uses *fast* mode: the ResNet50 backbone stays frozen,
every image goes through it once (the slow part — allow roughly 30–90 minutes for all 44,000 images on a laptop;
the result is cached, so re-runs take seconds), and then only the classification head is trained.
For a quick first trial use `python scripts/train_mri.py --max-per-class 300` (a few minutes).

With a graphics card the script switches to *finetune* mode, which also trains the last ResNet block and is the
mode to use for the highest accuracy. No graphics card? Upload the project to Google Colab, train there with
`python scripts/train_mri.py --mode finetune`, and copy `models/mri_resnet50.pt` back.

**Speech.** Takes a few seconds. Until you have DementiaBank it trains on the synthetic demo file — see
[The datasets](#the-datasets).

After training, your own accuracy figures, classification reports and confusion matrices are in `outputs/`.

### Using the app

1. Upload an MRI slice (any image from the dataset works).
2. Give a speech sample: record with the microphone, upload an audio file, or type the transcript.
   Ask the person to talk for about a minute, e.g. describing a picture or their morning.
3. Press **Analyse**. The first speech analysis downloads the Whisper model once (about 150 MB).

You can give only an MRI, or only speech; the app says which inputs the result is based on.

Command line instead of the app:

```
python scripts/predict.py --mri path\to\scan.jpg --audio path\to\recording.wav
python scripts/predict.py --mri path\to\scan.jpg --text "well uh the boy is um on the the thing"
```

---

## The datasets

Full details, links and citations are in [`datasets/README.md`](datasets/README.md).

| Modality | Dataset | In the zip? |
|---|---|---|
| MRI | Alzheimer's Disease Multiclass Images Dataset (Kaggle) — 44,000 images, 4 classes | No — fetched by `scripts/download_datasets.py` |
| Speech | DementiaBank Pitt Corpus (Cookie Theft descriptions) — Control / Dementia | No — access is by request only and it may not be redistributed |
| Speech (demo) | `datasets/speech/SYNTHETIC_demo_transcripts.csv` — 500 computer-generated transcripts | Yes |

The synthetic file exists only so that the speech pipeline runs before DementiaBank access arrives.
**It is not patient data, and accuracy measured on it is not a result.** The app, the training script and the
report all print a warning while the speech model in use was trained on it. Once the real transcripts are placed
in `datasets/speech/`, `python scripts/train_speech.py` picks them up automatically.

---

## How it works

```
 MRI image ─► resize 224×224, denoise ─► ResNet50 ─► 4-class probabilities ─┐
                                              └─► Grad-CAM ─► region summary │
                                                                             ├─► late fusion ─► risk score
 Speech ─► Whisper (speech-to-text) ─► transcript                            │        │
              └─ word timings          ├─► TF-IDF ─────────┐                 │        ▼
                 (rate, pauses)        └─► 8 markers ──────┴─► classifier ───┘   reasons + things to avoid
                                             └─► word-removal test                + HTML report
```

### MRI module — `src/mri/`
- **Preprocessing:** greyscale → 224×224 → 3×3 median filter (denoise) → 3 channels → ImageNet normalisation.
- **Model:** ResNet50 pretrained on ImageNet; the final layer is replaced by
  `LayerNorm → Dropout → Linear(2048→256) → ReLU → Dropout → Linear(256→4)`.
- **Split:** 80 % train / 20 % test, stratified. 10 % of the training part is used as a validation set to pick
  the best epoch, so the test set plays no part in any decision.
- **Grad-CAM** (`gradcam.py`): gradients of the predicted class with respect to the last convolutional block
  (`layer4`) weight its feature maps into a heat-map.
- **Region summary:** the heat-map is measured over three approximate zones of the slice — centre (where the
  ventricles are), outer edge (cortex), and the tissue between — plus left/right balance. If less than 60 % of the
  heat falls on brain tissue, the explanation is flagged as unreliable.

### Speech module — `src/speech/`
- **Speech-to-text** (`transcribe.py`): Whisper `base.en` on the CPU. It is prompted with a hesitant sentence so it
  keeps "uh" and "um" instead of tidying them away. Word timings give speaking rate and pauses of 2 s or longer.
- **Features** (`features.py`):
  - TF-IDF on the cleaned transcript (lower-case, stop-words removed, 1- and 2-word terms).
  - Eight linguistic markers: fillers, repetitions, vague words, pronouns, uncertainty phrases
    (all per 100 words), vocabulary variety (moving-average type/token ratio), content-word share, length.
- **Classifier** (`train.py`): Random Forest and Logistic Regression are both trained; the one with the better
  5-fold cross-validation accuracy is kept. 80 / 20 split.
- **Explanation** (`predict.py`):
  - *Words* — each word is removed in turn and the change in the dementia probability is recorded
    (works for any classifier, including Random Forest).
  - *Markers* — each marker is compared with the Control group of the training data and flagged when it is more
    than one standard deviation away in the direction of the Dementia group.

### Fusion — `src/fusion/fusion.py`
`risk = w_mri × (1 − P(NonDemented)) + w_speech × P(Dementia)`, with the weights proportional to each model's test
accuracy (0.6 / 0.4 while the speech model is the synthetic one; adjustable in the app).
Below 35 % = Low, above 65 % = High, otherwise Moderate. The system also reports whether the two results agree.

Late (decision-level) fusion is used because no public dataset has an MRI and a speech recording from the same
patients, so a joint model cannot be trained.

### Reasons and guidance — `src/xai/`
- `reasoning.py` turns the measurements into sentences with fixed rules — no language model, so the same input
  always gives the same explanation.
- `recommendations.py` holds the stage-wise "things to avoid / things to do / next steps" and its sources.

---

## Project layout

```
app.py                        web app (Streamlit)
config.py                     all paths and settings
requirements.txt
setup_windows.bat             one-time setup            run_app.bat   start the app
.vscode/                      Run-and-Debug entries and interpreter settings
scripts/
  download_datasets.py        fetch the MRI dataset, explain DementiaBank access
  train_mri.py                train the MRI model
  train_speech.py             train the speech model
  predict.py                  command-line analysis
  make_synthetic_speech_data.py   regenerates the synthetic demo transcripts
src/
  mri/      data.py  model.py  train.py  gradcam.py  predict.py
  speech/   data.py  features.py  train.py  transcribe.py  predict.py
  fusion/   fusion.py
  xai/      reasoning.py  recommendations.py
  pipeline.py                 runs everything in order
  report.py                   HTML report
datasets/                     see datasets/README.md
models/                       trained models are saved here
outputs/                      metrics, confusion matrices, reports
tests/smoke_test.py           installation check
```

---

## Differences from the presentation

| Presentation | This project | Why |
|---|---|---|
| Speech transcribed with the AssemblyAI API | Whisper running locally | Free, no API key, works offline. AssemblyAI is still supported: set the environment variable `ASSEMBLYAI_API_KEY` and run `pip install assemblyai`. |
| Text prediction labelled "Sentiment analysis: Positive / Negative" | Control / Dementia | The task is dementia detection, not sentiment; these are the DementiaBank labels. |
| Random Forest on one slide, Logistic Regression on another | Both are trained and the better one is kept | The chosen model and both cross-validation scores are saved in `outputs/speech_metrics.json`. |
| MRI and speech shown as two separate pipelines | Fused into one risk score with reasons and guidance | This is the "multimodal" objective. |

## Limitations

- The two models are trained on different patients; the fusion weights are a rule, not learned from paired data.
- The MRI dataset is an augmented collection. Augmented copies of one scan can fall on both sides of a random
  train/test split, so the test accuracy is optimistic compared with scans from new patients.
- The model sees single 2-D slices, not whole 3-D scans.
- Grad-CAM shows where the network looked. It is coarse (7×7 before upscaling) and is not a clinical finding;
  the zones in the written summary are geometric approximations.
- Speech is affected by many things besides dementia — language, education, hearing, tiredness, recording quality.
  The speech model is trained on English picture descriptions only.
- The guidance is general health information, not personal medical advice.

## References

1. He K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
2. Selvaraju R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
3. Radford A. et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). ICML.
4. Becker J. T. et al. (1994). The natural history of Alzheimer's disease. Archives of Neurology, 51(6), 585–594. (DementiaBank Pitt Corpus)
5. Fraser K. C., Meltzer J. A., Rudzicz F. (2016). Linguistic features identify Alzheimer's disease in narrative speech. Journal of Alzheimer's Disease, 49(2), 407–422.
6. World Health Organization (2019). Risk reduction of cognitive decline and dementia: WHO guidelines.
7. Livingston G. et al. (2024). Dementia prevention, intervention, and care: 2024 report of the Lancet standing Commission. The Lancet, 404, 572–628.
#   a l z h e i m e r s - m u l t i m o d a l - x a i  
 