"""Build a self-contained HTML report (images embedded) and a plain-text summary."""
from __future__ import annotations

import base64
import html
import io
import time

from PIL import Image

import config


def _img_tag(image: Image.Image, width: int = 240) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{data}" width="{width}" alt="">'


def _e(text) -> str:
    return html.escape(str(text))


def highlight_transcript(speech: dict, threshold: float = 0.01) -> str:
    """Transcript as HTML with influential words coloured (red = towards dementia, green = towards control)."""
    contrib = {w["word"]: w["contribution"] for w in speech["word_contributions"]}
    biggest = max((abs(v) for v in contrib.values()), default=0) or 1.0
    out = []
    for raw in speech["transcript"].split():
        key = "".join(ch for ch in raw.lower().replace("’", "'") if ch.isalpha() or ch == "'")
        c = contrib.get(key, 0.0)
        if abs(c) >= threshold:
            alpha = 0.15 + 0.55 * min(abs(c) / biggest, 1.0)
            colour = f"rgba(214,69,65,{alpha:.2f})" if c > 0 else f"rgba(39,174,96,{alpha:.2f})"
            out.append(f'<span style="background:{colour};border-radius:3px;padding:0 2px">{_e(raw)}</span>')
        else:
            out.append(_e(raw))
    return " ".join(out)


def text_summary(result: dict) -> str:
    f, g = result["fusion"], result["guidance"]
    lines = ["=" * 72, "MULTIMODAL ALZHEIMER'S ANALYSIS", "=" * 72,
             f"Overall risk: {f['risk_level']}  (score {f['risk_score']:.0%}, from {' + '.join(f['modalities_used'])})"]
    if result["mri"]:
        lines.append(f"MRI stage:    {result['mri']['predicted_label']}  ({result['mri']['confidence']:.0%} confidence)")
    if result["speech"]:
        lines.append(f"Speech:       {result['speech']['predicted_class']}  (P dementia {result['speech']['p_dementia']:.0%})")
        lines.append(f"Transcript:   {result['speech']['transcript']}")
    lines += ["", "WHY - reasons for this result", "-" * 72]
    lines += [f"[{r['source']}] {r['text']}" for r in result["reasons"]]
    lines += ["", f"GUIDANCE - {g['headline']}", "-" * 72, "Things to avoid:"]
    lines += [f"  x {a['item']} - {a['why']}" for a in g["avoid"] + g["avoid_communication"]]
    lines += ["General risk factors to avoid:"] + [f"  x {a['item']}" for a in g["avoid_general"]]
    lines += ["Things to do:"] + [f"  + {d}" for d in g["do"]]
    lines += ["Next steps:"] + [f"  > {n}" for n in g["next_steps"]]
    lines += ["", result["disclaimer"], "=" * 72]
    return "\n".join(lines)


def html_report(result: dict) -> str:
    f, g, mri, speech = result["fusion"], result["guidance"], result["mri"], result["speech"]
    colour = {"Low": "#1e8449", "Moderate": "#b9770e", "High": "#b03a2e"}[f["risk_level"]]
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Alzheimer's multimodal analysis report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:860px;margin:32px auto;padding:0 16px;color:#1c2833;line-height:1.5}}
h1{{font-size:22px}} h2{{font-size:17px;border-bottom:2px solid #d5d8dc;padding-bottom:4px;margin-top:28px}}
.risk{{border-left:8px solid {colour};background:#f8f9f9;padding:12px 16px;font-size:18px}}
.note{{background:#fef9e7;border:1px solid #f7dc6f;padding:10px 14px;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{border:1px solid #d5d8dc;padding:6px 8px;text-align:left}}
th{{background:#f2f4f4}} .imgs td{{border:none;text-align:center;font-size:12px}} li{{margin-bottom:5px}}
.tag{{display:inline-block;font-size:11px;background:#eaeded;border-radius:3px;padding:1px 6px;margin-right:6px}}
.transcript{{background:#f8f9f9;padding:12px;border-radius:4px}}
</style></head><body>
<h1>Multimodal Alzheimer's analysis report</h1>
<p>Generated {time.strftime('%d %b %Y, %H:%M')} &middot; inputs used: {_e(' + '.join(f['modalities_used']))}</p>
<div class="note"><b>Important:</b> {_e(result['disclaimer'])}</div>
<h2>1. Result</h2>
<div class="risk"><b style="color:{colour}">{f['risk_level']} risk</b> &mdash; combined score {f['risk_score']:.0%}<br>
<span style="font-size:14px">{_e(g['headline'])}</span></div>"""]

    if mri:
        rows = "".join(f"<tr><td>{_e(config.MRI_CLASS_LABELS[c])}</td><td>{p:.1%}</td></tr>"
                       for c, p in mri["probabilities"].items())
        parts.append(f"""<h2>2. MRI analysis (ResNet50 + Grad-CAM)</h2>
<p>Predicted stage: <b>{_e(mri['predicted_label'])}</b> ({mri['confidence']:.0%} confidence)</p>
<table class="imgs"><tr><td>{_img_tag(mri['preprocessed_image'])}<br>Preprocessed MRI</td>
<td>{_img_tag(mri['gradcam_overlay'])}<br>Grad-CAM (red = most influential)</td></tr></table>
<table><tr><th>Class</th><th>Probability</th></tr>{rows}</table>""")

    if speech:
        marker_rows = "".join(
            f"<tr><td>{_e(m['label'])}</td><td>{m['value']:.2f}</td>"
            f"<td>{m.get('control_mean', float('nan')):.2f}</td>"
            f"<td>{ {'towards_dementia': 'differs from healthy speech', 'towards_control': 'healthy', 'normal': 'usual range'}[m['flag']] }</td></tr>"
            for m in speech["markers"])
        parts.append(f"""<h2>3. Speech analysis (speech-to-text + NLP)</h2>
<p>Result: <b>{_e(speech['predicted_class'])}</b> (probability of dementia {speech['p_dementia']:.0%})
&middot; transcribed with: {_e(speech.get('engine', ''))} &middot; classifier: {_e(speech['model_meta'].get('model_name', ''))}</p>
<p class="transcript">{highlight_transcript(speech)}</p>
<p style="font-size:12px">Red words pushed the model towards Dementia, green words towards Control.</p>
<table><tr><th>Speech marker</th><th>This sample</th><th>Control average</th><th>Reading</th></tr>{marker_rows}</table>""")

    parts.append("<h2>4. Why the system gave this result</h2><ul>")
    parts += [f"<li><span class='tag'>{_e(r['source'])}</span>{_e(r['text'])}</li>" for r in result["reasons"]]
    parts.append("</ul><h2>5. Things to avoid</h2><ul>")
    parts += [f"<li><b>{_e(a['item'])}.</b> {_e(a['why'])}</li>" for a in g["avoid"] + g["avoid_communication"]]
    parts.append("</ul><p><b>Risk factors everyone should avoid</b></p><ul>")
    parts += [f"<li><b>{_e(a['item'])}.</b> {_e(a['why'])}</li>" for a in g["avoid_general"]]
    parts.append("</ul><h2>6. Things to do</h2><ul>")
    parts += [f"<li>{_e(d)}</li>" for d in g["do"]]
    parts.append("</ul><h2>7. Next steps</h2><ul>")
    parts += [f"<li>{_e(n)}</li>" for n in g["next_steps"]]
    parts.append("</ul><h2>Sources for the guidance</h2><ol style='font-size:12px'>")
    parts += [f"<li>{_e(s)}</li>" for s in g["sources"]]
    parts.append("</ol></body></html>")
    return "\n".join(parts)
