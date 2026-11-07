"""Load speech transcripts for training.

Supported inputs
  * a CSV / TSV file with one text column and one label column
    (column names are detected automatically)
  * a folder of DementiaBank CHAT files (*.cha) arranged as
        .../Control/.../*.cha   and   .../Dementia/.../*.cha
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import config

TEXT_COLUMNS = ["text", "transcript", "transcripts", "transcription", "utterance", "utterances",
                "speech", "sentence", "content", "description"]
LABEL_COLUMNS = ["label", "labels", "class", "category", "group", "diagnosis", "dx", "target", "type"]


def normalise_label(value) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    if key in {"1", "ad", "dementia", "demented", "probablead", "possiblead", "alzheimer",
               "alzheimers", "patient", "positive"}:
        return "Dementia"
    if key in {"0", "control", "controls", "hc", "healthy", "healthycontrol", "cn", "normal",
               "nondemented", "negative"}:
        return "Control"
    if "dement" in key or "alzheimer" in key:
        return "Dementia" if not key.startswith("non") else "Control"
    if "control" in key or "healthy" in key:
        return "Control"
    return None


def parse_cha(path: Path) -> str:
    """Extract the participant's (*PAR) speech from a CHAT transcript as plain text."""
    lines, current = [], None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.startswith("*"):
            if current is not None:
                lines.append(current)
            current = raw[5:].strip() if raw.startswith("*PAR:") else None
        elif raw.startswith("\t") and current is not None:
            current += " " + raw.strip()
        elif raw.startswith(("%", "@")):
            if current is not None:
                lines.append(current)
            current = None
    if current is not None:
        lines.append(current)

    text = " ".join(lines)
    text = re.sub(r"\x15.*?\x15", " ", text)              # time-alignment bullets
    text = re.sub(r"&[-=+]?(\w+)", r"\1", text)           # &-uh -> uh   (keep fillers)
    text = re.sub(r"\[[^\]]*\]", " ", text)               # [/] [//] [: word] [* error] ...
    text = re.sub(r"\((\w+)\)", r"\1", text)              # goin(g) -> going
    text = re.sub(r"\(\.+\)", " ", text)                  # (.) (..) pauses
    text = re.sub(r"\b(xxx|yyy|www)\b", " ", text)        # unintelligible
    text = re.sub(r"[<>+/\"@]|\b0\w+", " ", text)
    text = re.sub(r"\s+([.?!,])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    lower = {c.lower().strip(): c for c in df.columns}
    text_col = next((lower[c] for c in TEXT_COLUMNS if c in lower), None)
    label_col = next((lower[c] for c in LABEL_COLUMNS if c in lower), None)
    if text_col is None or label_col is None:
        raise SystemExit(
            f"{path.name}: could not find the text/label columns. Found columns {list(df.columns)}.\n"
            f"Rename them to 'text' and 'label' (label values: Control / Dementia)."
        )
    out = pd.DataFrame({"text": df[text_col].astype(str), "label": df[label_col].map(normalise_label)})
    out["source"] = df[lower["source"]].astype(str) if "source" in lower else path.name
    unknown = out["label"].isna().sum()
    if unknown:
        print(f"[warning] {unknown} rows in {path.name} have a label that is neither Control nor Dementia - skipped.")
    return out.dropna(subset=["label"])


def _load_cha_folder(folder: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.rglob("*.cha")):
        label = next((normalise_label(p.name) for p in path.relative_to(folder).parents if p.name
                      and normalise_label(p.name)), None)
        if label:
            text = parse_cha(path)
            if text:
                rows.append({"text": text, "label": label, "source": "DementiaBank"})
    return pd.DataFrame(rows)


def find_default_dataset() -> Path:
    """Prefer real data placed in datasets/speech; fall back to the synthetic demo file."""
    folder = config.SPEECH_DATA_DIR
    real_tables = [p for p in sorted(folder.glob("*.[ct]sv")) if not p.name.upper().startswith("SYNTHETIC")]
    if real_tables:
        return real_tables[0]
    if any(folder.rglob("*.cha")):
        return folder
    return config.SYNTHETIC_SPEECH_CSV


def load_transcripts(path: Path | None = None) -> tuple[pd.DataFrame, bool]:
    """Return (DataFrame[text, label, source], is_synthetic)."""
    path = Path(path) if path else find_default_dataset()
    if not path.exists():
        raise SystemExit(f"Speech dataset not found: {path}\nSee datasets/README.md.")
    df = _load_cha_folder(path) if path.is_dir() else _load_table(path)
    if df.empty:
        raise SystemExit(f"No usable transcripts found in {path}.")
    df = df[df["text"].str.split().str.len() >= 5].reset_index(drop=True)
    is_synthetic = bool((df["source"].str.lower() == "synthetic").all()) or path.name.upper().startswith("SYNTHETIC")
    return df, is_synthetic
