"""Command-line analysis.

Examples
  python scripts/predict.py --mri path/to/scan.jpg
  python scripts/predict.py --audio path/to/recording.wav
  python scripts/predict.py --mri scan.jpg --text "well uh the boy is um on the the thing"
  python scripts/predict.py --mri scan.jpg --audio rec.wav --report outputs/report.html
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.pipeline import analyse  # noqa: E402
from src.report import html_report, text_summary  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyse an MRI and/or a speech sample and explain the result.")
    ap.add_argument("--mri", type=Path, help="MRI image file")
    ap.add_argument("--audio", type=Path, help="audio file with the person's speech")
    ap.add_argument("--text", type=str, help="transcript, if you already have one (skips speech-to-text)")
    ap.add_argument("--report", type=Path, default=config.OUTPUTS_DIR / "report.html", help="where to save the HTML report")
    args = ap.parse_args()
    if not (args.mri or args.audio or args.text):
        ap.error("give at least one of --mri, --audio, --text")

    result = analyse(mri_image=args.mri, audio=args.audio, text=args.text)
    print(text_summary(result))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(html_report(result), encoding="utf-8")
    print(f"\nHTML report saved to {args.report}")
    if result["mri"]:
        overlay = args.report.with_name("gradcam_overlay.png")
        result["mri"]["gradcam_overlay"].save(overlay)
        print(f"Grad-CAM image saved to {overlay}")
