"""Text cleaning and hand-crafted linguistic markers for dementia speech.

Besides *which* words a person uses (TF-IDF), research on dementia speech
shows that *how* they speak carries a lot of signal: more fillers and
repetitions, vaguer words, more pronouns in place of nouns, and a smaller
vocabulary. The markers below measure exactly those things, and because each
one has a plain meaning they can be shown to the user as reasons.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

FILLERS = {"uh", "um", "er", "ah", "eh", "hm", "hmm", "mm", "mhm", "erm", "uhm", "umm", "uhh"}
PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their",
            "this", "that", "these", "those", "there", "here"}
VAGUE_WORDS = {"thing", "things", "stuff", "something", "someone", "somebody", "somewhere",
               "anything", "everything", "whatever", "thingy", "whatsit"}
UNCERTAIN_PHRASES = ["i don't know", "i dont know", "i do not know", "i can't remember",
                     "i cant remember", "i cannot remember", "i don't remember", "i dont remember",
                     "i forget", "i forgot", "what's it called", "whats it called",
                     "what do you call", "i can't think", "i cant think", "i'm not sure", "im not sure",
                     "i can't see", "i cant see", "i can't tell", "i cant tell"]

# name, plain-language label, unit
MARKERS = [
    ("filler_rate", "Filler sounds (uh, um, er)", "per 100 words"),
    ("repetition_rate", "Repeated words and phrases", "per 100 words"),
    ("vague_rate", "Vague words (thing, stuff, something)", "per 100 words"),
    ("pronoun_rate", "Pronouns used in place of names (it, that, there)", "per 100 words"),
    ("uncertainty_rate", "Phrases of uncertainty (\"I don't know\", \"I can't remember\")", "per 100 words"),
    ("lexical_diversity", "Vocabulary variety", "0-1, higher = richer"),
    ("content_ratio", "Share of meaningful content words", "0-1, higher = more informative"),
    ("word_count", "Length of the description", "words"),
]
MARKER_NAMES = [m[0] for m in MARKERS]
MARKER_LABELS = {m[0]: m[1] for m in MARKERS}
MARKER_UNITS = {m[0]: m[2] for m in MARKERS}

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean_text(text: str) -> str:
    """Lower-case and keep only words - used before TF-IDF."""
    return " ".join(tokenize(text))


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower().replace("’", "'"))


def _moving_ttr(tokens: list[str], window: int = 25) -> float:
    """Moving-average type/token ratio - a vocabulary measure that does not
    automatically drop just because a text is longer."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]
    return float(np.mean(ratios))


def _repetitions(tokens: list[str]) -> int:
    """Immediate word repeats ("the the") and immediate two-word repeats ("in the in the")."""
    count = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1] and tokens[i] not in FILLERS:
            count += 1
    for i in range(3, len(tokens)):
        if tokens[i - 1:i + 1] == tokens[i - 3:i - 1] and tokens[i] != tokens[i - 1]:
            count += 1
    return count


def linguistic_markers(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    n = len(tokens)
    if n == 0:
        return {name: 0.0 for name in MARKER_NAMES}
    joined = " ".join(tokens)
    words_only = [t for t in tokens if t not in FILLERS]
    per100 = 100.0 / n
    return {
        "filler_rate": sum(t in FILLERS for t in tokens) * per100,
        "repetition_rate": _repetitions(tokens) * per100,
        "vague_rate": sum(t in VAGUE_WORDS for t in tokens) * per100,
        "pronoun_rate": sum(t in PRONOUNS for t in tokens) * per100,
        "uncertainty_rate": sum(joined.count(p) for p in UNCERTAIN_PHRASES) * per100,
        "lexical_diversity": _moving_ttr(words_only),
        "content_ratio": sum(t not in ENGLISH_STOP_WORDS and t not in FILLERS for t in tokens) / n,
        "word_count": float(n),
    }


class LinguisticFeatures(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: list of texts -> matrix of linguistic markers."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            m = linguistic_markers(text)
            row = [m[name] for name in MARKER_NAMES]
            row[-1] = np.log1p(row[-1])          # word count: compress the long tail
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(MARKER_NAMES)
