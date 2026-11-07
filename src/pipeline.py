"""One entry point that runs everything: MRI -> speech -> fusion -> reasons -> guidance."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image

import config
from src.fusion.fusion import fuse
from src.xai.reasoning import explain_all
from src.xai.recommendations import build_guidance


@lru_cache(maxsize=1)
def get_mri_predictor():
    from src.mri.predict import MRIPredictor
    return MRIPredictor()


@lru_cache(maxsize=1)
def get_speech_predictor():
    from src.speech.predict import SpeechPredictor
    return SpeechPredictor()


def models_available() -> dict:
    return {"mri": config.MRI_MODEL_PATH.exists(), "speech": config.SPEECH_MODEL_PATH.exists()}


def analyse(mri_image: Image.Image | str | Path | None = None,
            audio: str | Path | bytes | None = None,
            text: str | None = None,
            w_mri: float | None = None, w_speech: float | None = None,
            whisper_size: str = config.WHISPER_MODEL_SIZE) -> dict:
    """Run the full analysis. Give an MRI image, and/or audio or an already-typed transcript."""
    mri_result = speech_result = transcription = None

    if mri_image is not None:
        mri_result = get_mri_predictor().predict(mri_image)

    if audio is not None and not (text and text.strip()):
        from src.speech.transcribe import transcribe
        transcription = transcribe(audio, model_size=whisper_size)
        text = transcription["text"]

    if text and text.strip():
        timing = transcription["timing"] if transcription else None
        speech_result = get_speech_predictor().predict(text.strip(), timing=timing)
        speech_result["engine"] = transcription["engine"] if transcription else "typed text"
    elif audio is not None:
        raise ValueError("No speech could be recognised in the recording. Try a clearer or longer recording.")

    if mri_result is None and speech_result is None:
        raise ValueError("Nothing to analyse: provide an MRI image and/or a speech sample.")

    fusion = fuse(mri_result, speech_result, w_mri, w_speech)
    return {
        "mri": mri_result,
        "speech": speech_result,
        "fusion": fusion,
        "reasons": explain_all(mri_result, speech_result, fusion),
        "guidance": build_guidance(mri_result, speech_result, fusion),
        "disclaimer": config.DISCLAIMER,
    }
