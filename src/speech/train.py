"""Train the speech (transcript) classifier: Control vs Dementia.

Features  TF-IDF of the cleaned transcript (unigrams + bigrams, stop-words removed)
          + eight hand-crafted linguistic markers (see features.py)
Models    Logistic Regression and Random Forest are both trained; the one with the
          better 5-fold cross-validation accuracy on the training part is kept.
Split     80 % train / 20 % test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.speech.data import load_transcripts
from src.speech.features import MARKER_NAMES, LinguisticFeatures, clean_text, linguistic_markers


def build_pipeline(classifier) -> Pipeline:
    features = FeatureUnion([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english", ngram_range=(1, 2),
                                  min_df=2, max_features=5000, sublinear_tf=True)),
        ("markers", Pipeline([("extract", LinguisticFeatures()), ("scale", StandardScaler())])),
    ])
    return Pipeline([("features", features), ("clf", classifier)])


def candidates(seed: int) -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced", random_state=seed),
    }


def _reference_stats(texts, labels) -> dict:
    """Mean / std of every marker per class - the yardstick used in explanations."""
    stats = {}
    labels = np.asarray(labels)
    table = np.asarray([[linguistic_markers(t)[m] for m in MARKER_NAMES] for t in texts])
    for cls in config.SPEECH_CLASSES:
        rows = table[labels == cls]
        stats[cls] = {m: {"mean": float(rows[:, i].mean()), "std": float(rows[:, i].std() + 1e-6)}
                      for i, m in enumerate(MARKER_NAMES)}
    return stats


def _save_confusion_matrix(cm, classes, path: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Purples")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Speech Classification Confusion Matrix")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(data_path: Path | None = None, model: str = "auto", seed: int = config.RANDOM_SEED,
          model_path: Path = config.SPEECH_MODEL_PATH, output_dir: Path = config.OUTPUTS_DIR) -> dict:
    df, is_synthetic = load_transcripts(data_path)
    counts = df["label"].value_counts().to_dict()
    print(f"Transcripts: {len(df)}  {counts}")
    if is_synthetic:
        print("\n" + "!" * 78)
        print("  You are training on the SYNTHETIC demo transcripts. They are computer-generated")
        print("  so the pipeline can run end to end. The accuracy below says nothing about")
        print("  real patients. Use DementiaBank for real results (see datasets/README.md).")
        print("!" * 78 + "\n")
    if len(counts) < 2 or min(counts.values()) < 10:
        raise SystemExit("Need at least 10 transcripts in each of the two classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(), test_size=config.TEST_SPLIT,
        stratify=df["label"], random_state=seed)

    options = candidates(seed)
    if model in ("rf", "random_forest"):
        options = {"Random Forest": options["Random Forest"]}
    elif model in ("lr", "logistic_regression"):
        options = {"Logistic Regression": options["Logistic Regression"]}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = {}
    for name, clf in options.items():
        scores = cross_val_score(build_pipeline(clf), x_train, y_train, cv=cv, scoring="accuracy")
        cv_scores[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        print(f"  {name:20s} 5-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    best_name = max(cv_scores, key=lambda n: cv_scores[n]["mean"])
    print(f"Selected model: {best_name}")

    pipeline = build_pipeline(options[best_name]).fit(x_train, y_train)
    preds = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, preds))
    report_text = classification_report(y_test, preds, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=config.SPEECH_CLASSES)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrix(cm, config.SPEECH_CLASSES, output_dir / "speech_confusion_matrix.png")
    (output_dir / "speech_classification_report.txt").write_text(report_text, encoding="utf-8")

    meta = {
        "test_accuracy": accuracy,
        "model_name": best_name,
        "cv_scores": cv_scores,
        "n_transcripts": len(df),
        "class_counts": counts,
        "n_test": len(y_test),
        "is_synthetic": is_synthetic,
        "trained_on": time.strftime("%Y-%m-%d %H:%M"),
        "reference_stats": _reference_stats(x_train, y_train),
    }
    (output_dir / "speech_metrics.json").write_text(json.dumps(
        {**meta, "report": classification_report(y_test, preds, output_dict=True, zero_division=0),
         "confusion_matrix": cm.tolist()}, indent=2), encoding="utf-8")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "meta": meta}, model_path)

    print("\n" + report_text)
    print(f"Test accuracy: {accuracy:.4f}" + ("   (synthetic data - not a real result)" if is_synthetic else ""))
    print(f"Model saved to   {model_path}")
    print(f"Reports saved to {output_dir}")
    return meta
