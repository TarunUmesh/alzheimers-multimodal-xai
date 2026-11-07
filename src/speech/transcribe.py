"""Speech-to-text.

Default engine: Whisper, running locally through the `faster-whisper` package.
  * free, no API key, works offline after the model is downloaded once (~150 MB)
  * reads wav / mp3 / m4a / ogg / webm directly (no separate ffmpeg install)

Optional engine: AssemblyAI (the cloud API named in the original methodology).
  It is used only if the environment variable ASSEMBLYAI_API_KEY is set and
  `pip install assemblyai` has been run.

Both return the text plus word timings, from which speaking rate and long
pauses are measured.
"""
from __future__ import annotations

import io
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import config

# Whisper tends to "tidy up" speech and drop hesitations. Starting it off with a
# prompt that itself contains hesitations makes it keep them, which matters here
# because fillers and repetitions are exactly what we want to measure.
DISFLUENCY_PROMPT = "Umm, let me think, uh... well, I, I see the, the thing there. Hmm, er, I don't know."

LONG_PAUSE_SECONDS = 2.0


@lru_cache(maxsize=2)
def _whisper(model_size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _timing_stats(words: list[dict], duration: float) -> dict:
    if not words:
        return {"duration_seconds": duration, "words_per_minute": 0.0, "long_pauses": 0, "longest_pause": 0.0}
    gaps = [b["start"] - a["end"] for a, b in zip(words, words[1:])]
    spoken_span = max(words[-1]["end"] - words[0]["start"], 1e-6)
    return {
        "duration_seconds": float(duration),
        "words_per_minute": float(len(words) / spoken_span * 60.0),
        "long_pauses": int(sum(g >= LONG_PAUSE_SECONDS for g in gaps)),
        "longest_pause": float(max(gaps, default=0.0)),
    }


def _transcribe_whisper(audio, model_size: str) -> dict:
    model = _whisper(model_size)
    segments, info = model.transcribe(audio, language="en", word_timestamps=True, beam_size=5,
                                      initial_prompt=DISFLUENCY_PROMPT,
                                      condition_on_previous_text=False)
    words, parts = [], []
    for seg in segments:
        parts.append(seg.text.strip())
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
    return {"text": " ".join(parts).strip(), "words": words,
            "timing": _timing_stats(words, float(info.duration)), "engine": f"whisper ({model_size})"}


def _transcribe_assemblyai(path: str) -> dict:
    import assemblyai as aai
    aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]
    transcript = aai.Transcriber().transcribe(path, config=aai.TranscriptionConfig(disfluencies=True))
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI error: {transcript.error}")
    words = [{"word": w.text, "start": w.start / 1000.0, "end": w.end / 1000.0} for w in (transcript.words or [])]
    duration = float(transcript.audio_duration or (words[-1]["end"] if words else 0.0))
    return {"text": transcript.text or "", "words": words,
            "timing": _timing_stats(words, duration), "engine": "AssemblyAI"}


def transcribe(audio: str | Path | bytes, model_size: str = config.WHISPER_MODEL_SIZE,
               engine: str = "auto") -> dict:
    """Transcribe an audio file (path) or raw audio bytes.

    Returns {"text", "words": [{"word","start","end"}], "timing": {...}, "engine"}.
    """
    use_assembly = engine == "assemblyai" or (engine == "auto" and os.environ.get("ASSEMBLYAI_API_KEY"))
    if use_assembly:
        try:
            if isinstance(audio, bytes):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio)
                try:
                    return _transcribe_assemblyai(tmp.name)
                finally:
                    os.unlink(tmp.name)
            return _transcribe_assemblyai(str(audio))
        except ImportError:
            if engine == "assemblyai":
                raise RuntimeError("Run  pip install assemblyai  to use the AssemblyAI engine.")
            print("[info] assemblyai package not installed - using local Whisper instead.")

    source = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
    return _transcribe_whisper(source, model_size)
