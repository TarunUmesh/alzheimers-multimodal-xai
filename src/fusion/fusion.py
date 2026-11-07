"""Decision-level (late) fusion of the MRI and speech results.

Why late fusion?  There is no public dataset in which the *same* patients have
both an MRI scan and a speech recording, so a joint model cannot be trained.
Instead each modality is trained on its own dataset and their outputs are
combined afterwards:

    risk = w_mri * P(MRI shows impairment) + w_speech * P(speech shows dementia)

where  P(MRI shows impairment) = 1 - P(NonDemented).
The weights are proportional to each model's test accuracy, so the more
reliable model has the bigger say. They can be overridden by the user.
"""
from __future__ import annotations

import config


def default_weights(mri: dict | None, speech: dict | None) -> tuple[float, float]:
    mri_acc = (mri or {}).get("model_meta", {}).get("test_accuracy")
    speech_meta = (speech or {}).get("model_meta", {})
    speech_acc = speech_meta.get("test_accuracy")
    # Accuracy measured on synthetic transcripts says nothing about real speech,
    # so it must not be allowed to set the weights.
    if mri_acc and speech_acc and not speech_meta.get("is_synthetic"):
        total = mri_acc + speech_acc
        return mri_acc / total, speech_acc / total
    return config.DEFAULT_MRI_WEIGHT, config.DEFAULT_SPEECH_WEIGHT


def risk_level(score: float) -> str:
    if score < config.RISK_LOW_BELOW:
        return "Low"
    if score > config.RISK_HIGH_ABOVE:
        return "High"
    return "Moderate"


def fuse(mri: dict | None = None, speech: dict | None = None,
         w_mri: float | None = None, w_speech: float | None = None) -> dict:
    if mri is None and speech is None:
        raise ValueError("Give at least one of: MRI result, speech result.")

    if mri is not None and speech is not None:
        if w_mri is None or w_speech is None:
            w_mri, w_speech = default_weights(mri, speech)
        total = w_mri + w_speech
        w_mri, w_speech = w_mri / total, w_speech / total
        p_mri, p_speech = mri["p_impaired"], speech["p_dementia"]
        score = w_mri * p_mri + w_speech * p_speech
        mri_positive, speech_positive = p_mri >= 0.5, p_speech >= 0.5
        if mri_positive and speech_positive:
            agreement = "both_positive"
        elif not mri_positive and not speech_positive:
            agreement = "both_negative"
        else:
            agreement = "conflict"
        used = ["MRI", "Speech"]
    elif mri is not None:
        w_mri, w_speech, score, agreement, used = 1.0, 0.0, mri["p_impaired"], "single", ["MRI"]
    else:
        w_mri, w_speech, score, agreement, used = 0.0, 1.0, speech["p_dementia"], "single", ["Speech"]

    return {
        "risk_score": float(score),
        "risk_level": risk_level(score),
        "agreement": agreement,
        "modalities_used": used,
        "weights": {"mri": float(w_mri), "speech": float(w_speech)},
        "p_mri_impaired": None if mri is None else float(mri["p_impaired"]),
        "p_speech_dementia": None if speech is None else float(speech["p_dementia"]),
        "mri_stage": None if mri is None else mri["predicted_class"],
    }
