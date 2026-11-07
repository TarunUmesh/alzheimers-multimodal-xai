"""Classify a transcript and explain the decision."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np

import config
from src.speech.features import (FILLERS, MARKER_LABELS, MARKER_NAMES, MARKER_UNITS,
                                 linguistic_markers, tokenize)

MIN_WORDS_RELIABLE = 30


class SpeechPredictor:
    def __init__(self, model_path: Path = config.SPEECH_MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained speech model at {model_path}. Train one first with:  python scripts/train_speech.py"
            )
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.meta = bundle["meta"]
        self._dementia_col = list(self.pipeline.classes_).index("Dementia")

    def _p_dementia(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)[:, self._dementia_col]

    # ---------------------------------------------------------------- words
    def word_contributions(self, text: str, max_words: int = 80) -> list[dict]:
        """Occlusion test: delete one word everywhere and see how the probability moves.

        contribution > 0  -> the word pushed the model TOWARDS 'Dementia'
        contribution < 0  -> the word pushed the model TOWARDS 'Control'
        It works for any classifier, which is why it is used instead of reading
        model coefficients (a Random Forest has none).
        """
        tokens = tokenize(text)
        if not tokens:
            return []
        vocab = [w for w, _ in Counter(tokens).most_common(max_words)]
        variants = [" ".join(t for t in tokens if t != w) for w in vocab]
        base = float(self._p_dementia([" ".join(tokens)])[0])
        probs = self._p_dementia(variants)
        counts = Counter(tokens)
        out = [{"word": w, "contribution": base - float(p), "count": counts[w]} for w, p in zip(vocab, probs)]
        return sorted(out, key=lambda d: abs(d["contribution"]), reverse=True)

    # -------------------------------------------------------------- markers
    def marker_findings(self, text: str) -> list[dict]:
        """Compare every linguistic marker with the Control group of the training data."""
        values = linguistic_markers(text)
        ref = self.meta.get("reference_stats", {})
        findings = []
        for name in MARKER_NAMES:
            item = {"name": name, "label": MARKER_LABELS[name], "unit": MARKER_UNITS[name],
                    "value": values[name], "flag": "normal"}
            if ref:
                control, dementia = ref["Control"][name], ref["Dementia"][name]
                z = (values[name] - control["mean"]) / control["std"]
                dementia_direction = np.sign(dementia["mean"] - control["mean"])
                item.update({"control_mean": control["mean"], "dementia_mean": dementia["mean"], "z": float(z)})
                if z * dementia_direction >= 1.0:
                    item["flag"] = "towards_dementia"
                elif z * dementia_direction <= -1.0:
                    item["flag"] = "towards_control"
            findings.append(item)
        return findings

    # -------------------------------------------------------------- predict
    def predict(self, text: str, timing: dict | None = None) -> dict:
        tokens = tokenize(text)
        p = float(self._p_dementia([text])[0]) if tokens else 0.5
        label = "Dementia" if p >= 0.5 else "Control"
        n_words = len([t for t in tokens if t not in FILLERS])
        return {
            "transcript": text,
            "predicted_class": label,
            "p_dementia": p,
            "confidence": p if label == "Dementia" else 1 - p,
            "word_count": len(tokens),
            "too_short": n_words < MIN_WORDS_RELIABLE,
            "word_contributions": self.word_contributions(text),
            "markers": self.marker_findings(text),
            "timing": timing,
            "model_meta": {k: v for k, v in self.meta.items() if k != "reference_stats"},
        }
