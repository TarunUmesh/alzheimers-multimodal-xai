"""Installation check - runs the WHOLE pipeline in about a minute, no dataset needed.

    python tests/smoke_test.py

It draws crude fake "scans" (grey ovals with a dark centre), trains both models
on them for a few seconds in a temporary folder, and then runs prediction,
Grad-CAM, fusion, explanation and report generation. Nothing in models/ or
datasets/ is touched. The accuracy it prints is meaningless; the only question
it answers is "does every part of the code run on this computer?".
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import config  # noqa: E402


def fake_scan(rng, severity: int) -> Image.Image:
    img = Image.new("L", (176, 208), 0)
    d = ImageDraw.Draw(img)
    d.ellipse([28, 24, 148, 184], fill=int(rng.integers(140, 200)))
    r = 8 + 7 * severity + int(rng.integers(0, 4))          # "ventricles" grow with severity
    d.ellipse([78 - r, 104 - 2 * r, 84, 104 + 2 * r], fill=20)
    d.ellipse([92, 104 - 2 * r, 98 + r, 104 + 2 * r], fill=20)
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 6, (208, 176))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    from scripts.make_synthetic_speech_data import make_transcript
    from src.fusion.fusion import fuse
    from src.mri.predict import MRIPredictor
    from src.mri.train import train as train_mri
    from src.report import html_report, text_summary
    from src.speech.predict import SpeechPredictor
    from src.speech.train import train as train_speech
    from src.xai.reasoning import explain_all
    from src.xai.recommendations import build_guidance
    import csv
    import random

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("[1/5] creating fake scans")
        for severity, cls in enumerate(config.MRI_CLASSES):
            (tmp / "mri" / cls).mkdir(parents=True)
            for i in range(16):
                fake_scan(rng, severity).save(tmp / "mri" / cls / f"{i}.png")

        print("[2/5] training the MRI model (tiny run, random starting weights)")
        train_mri(data_dir=tmp / "mri", mode="fast", epochs=3, batch_size=16, pretrained=False,
                  model_path=tmp / "mri.pt", output_dir=tmp / "out")

        print("[3/5] training the speech model on synthetic transcripts")
        r = random.Random(0)
        with open(tmp / "SYNTHETIC_speech.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text", "label", "source"])
            for label in config.SPEECH_CLASSES:
                for _ in range(60):
                    w.writerow([make_transcript(r, label), label, "synthetic"])
        train_speech(data_path=tmp / "SYNTHETIC_speech.csv", model_path=tmp / "speech.joblib",
                     output_dir=tmp / "out")

        print("[4/5] predicting + Grad-CAM + explanation")
        mri = MRIPredictor(tmp / "mri.pt").predict(tmp / "mri" / "ModerateDemented" / "0.png")
        speech = SpeechPredictor(tmp / "speech.joblib").predict(
            "Well uh the the boy is um up on that thing. He is getting something. I don't know. "
            "Uh the lady is doing the things and the water is going. I can't remember what it's called.")
        assert mri["gradcam"].shape == (config.IMAGE_SIZE, config.IMAGE_SIZE)
        assert abs(sum(mri["probabilities"].values()) - 1) < 1e-4
        assert 0 <= speech["p_dementia"] <= 1

        print("[5/5] fusion + guidance + report")
        fusion = fuse(mri, speech)
        result = {"mri": mri, "speech": speech, "fusion": fusion,
                  "reasons": explain_all(mri, speech, fusion),
                  "guidance": build_guidance(mri, speech, fusion), "disclaimer": config.DISCLAIMER}
        assert len(html_report(result)) > 1000
        print(text_summary(result))

    print("\nSMOKE TEST PASSED - every part of the pipeline runs on this computer.")


if __name__ == "__main__":
    main()
