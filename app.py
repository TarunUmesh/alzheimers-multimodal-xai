"""Web app.   Start it with:   streamlit run app.py

Upload a brain MRI slice and/or record (or upload, or type) a speech sample.
The app shows the prediction of each model, the combined risk, the reasons
behind it, and stage-wise guidance on what to avoid.
"""
import pandas as pd
import streamlit as st
from PIL import Image

import config
from src import pipeline
from src.report import highlight_transcript, html_report

st.set_page_config(page_title="Alzheimer's Multimodal Explainable AI", page_icon="-", layout="wide")

RISK_COLOUR = {"Low": "#1e8449", "Moderate": "#b9770e", "High": "#b03a2e"}
KIND_ICON = {"prediction": "🔹", "evidence": "🔎", "caution": "⚠️", "note": "ℹ️"}
SPEAKING_TASK = ("Ask the person to speak for about one minute - for example to describe a detailed picture, "
                 "or to describe everything they did this morning from waking up. Do not prompt or help them.")


# ------------------------------------------------------------------ sidebar
def sidebar() -> dict:
    available = pipeline.models_available()
    with st.sidebar:
        st.header("Models")
        if available["mri"]:
            meta = pipeline.get_mri_predictor().meta
            st.success(f"MRI model ready\n\nResNet50 · test accuracy {meta.get('test_accuracy', 0):.1%} "
                       f"· {meta.get('n_images', '?')} images · mode: {meta.get('mode', '?')}")
        else:
            st.error("MRI model not trained yet.\n\nRun `python scripts/train_mri.py`")
        if available["speech"]:
            meta = pipeline.get_speech_predictor().meta
            st.success(f"Speech model ready\n\n{meta.get('model_name')} · test accuracy "
                       f"{meta.get('test_accuracy', 0):.1%} · {meta.get('n_transcripts')} transcripts")
            if meta.get("is_synthetic"):
                st.warning("The speech model was trained on **synthetic demo transcripts**. "
                           "Retrain on DementiaBank for real results.")
        else:
            st.error("Speech model not trained yet.\n\nRun `python scripts/train_speech.py`")

        st.header("Settings")
        manual = st.toggle("Set fusion weights by hand", value=False,
                           help="By default the weights follow each model's test accuracy.")
        w_mri = st.slider("Weight of MRI result", 0.0, 1.0, config.DEFAULT_MRI_WEIGHT, 0.05) if manual else None
        whisper = st.selectbox("Speech-to-text model", ["tiny.en", "base.en", "small.en"],
                               index=["tiny.en", "base.en", "small.en"].index(config.WHISPER_MODEL_SIZE),
                               help="Larger = more accurate but slower. Downloaded once on first use.")
        st.divider()
        st.caption(config.DISCLAIMER)
    return {"available": available, "w_mri": w_mri, "whisper": whisper}


# ------------------------------------------------------------------- inputs
def inputs(available: dict):
    left, right = st.columns(2, gap="large")
    image = audio = text = None

    with left:
        st.subheader("1 · Brain MRI")
        file = st.file_uploader("Upload an MRI slice (PNG / JPG)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                                disabled=not available["mri"])
        if file:
            image = Image.open(file)
            st.image(image, caption="Uploaded MRI", width=260)

    with right:
        st.subheader("2 · Speech sample")
        st.caption(SPEAKING_TASK)
        mode = st.radio("How do you want to give the speech?",
                        ["Record with microphone", "Upload audio file", "Type or paste transcript"],
                        horizontal=True, disabled=not available["speech"])
        if mode == "Record with microphone":
            recording = st.audio_input("Press the microphone, speak, press again to stop",
                                       disabled=not available["speech"])
            if recording:
                audio = recording.getvalue()
        elif mode == "Upload audio file":
            upload = st.file_uploader("Upload audio (WAV / MP3 / M4A / OGG / WEBM)",
                                      type=["wav", "mp3", "m4a", "ogg", "webm", "flac"])
            if upload:
                audio = upload.getvalue()
                st.audio(audio)
        else:
            text = st.text_area("Transcript of what the person said", height=170,
                                placeholder="Type exactly what was said, including 'uh', 'um' and repeated words.")
    return image, audio, text


# ------------------------------------------------------------------ results
def show_overview(result: dict):
    fusion, guidance = result["fusion"], result["guidance"]
    colour = RISK_COLOUR[fusion["risk_level"]]
    st.markdown(
        f"<div style='border-left:10px solid {colour};background:rgba(128,128,128,0.10);padding:14px 18px;"
        f"border-radius:4px'><span style='font-size:26px;font-weight:700;color:{colour}'>"
        f"{fusion['risk_level']} risk</span> &nbsp; combined score {fusion['risk_score']:.0%}"
        f"<br>{guidance['headline']}</div>", unsafe_allow_html=True)
    st.write("")
    agreement = {"both_positive": "Both show signs", "both_negative": "Both look healthy",
                 "conflict": "Results disagree", "single": "One input only"}[fusion["agreement"]]
    mri, speech = result["mri"], result["speech"]
    cards = [
        ("MRI stage", mri["predicted_label"] if mri else "not provided",
         f"{mri['confidence']:.0%} confidence" if mri else ""),
        ("Speech", speech["predicted_class"] if speech else "not provided",
         f"probability of dementia {speech['p_dementia']:.0%}" if speech else ""),
        ("Agreement", agreement,
         f"weights: MRI {fusion['weights']['mri']:.2f} / speech {fusion['weights']['speech']:.2f}"),
    ]
    for column, (label, value, detail) in zip(st.columns(3), cards):
        column.markdown(f"<div style='font-size:13px;opacity:0.7'>{label}</div>"
                        f"<div style='font-size:22px;font-weight:600'>{value}</div>"
                        f"<div style='font-size:13px;opacity:0.7'>{detail}</div>", unsafe_allow_html=True)

    st.subheader("Why the system gave this result")
    for source in ["Combined", "MRI", "Speech"]:
        items = [r for r in result["reasons"] if r["source"] == source]
        if items:
            st.markdown(f"**{source}**")
            for r in items:
                st.markdown(f"{KIND_ICON.get(r['kind'], '•')} {r['text']}")


def show_mri(mri: dict):
    c1, c2, c3 = st.columns([1, 1, 1.2])
    c1.image(mri["preprocessed_image"], caption="Preprocessed MRI (what the model sees)", width=280)
    c2.image(mri["gradcam_overlay"], caption="Grad-CAM: red = most influential", width=280)
    with c3:
        st.markdown(f"**Predicted stage: {mri['predicted_label']}**")
        for cls, p in mri["probabilities"].items():
            st.progress(float(p), text=f"{config.MRI_CLASS_LABELS[cls]} - {p:.1%}")
    zones = (mri.get("regions") or {}).get("zones")
    if zones:
        st.markdown("**Where the model looked** (share of heat-map vs share of brain area)")
        names = {"central": "Centre (ventricles)", "deep_middle": "Between centre and edge",
                 "outer_cortex": "Outer edge (cortex)", "image_left": "Left half of image",
                 "image_right": "Right half of image", "front_half": "Top half of image",
                 "back_half": "Bottom half of image"}
        st.dataframe(pd.DataFrame([
            {"Image zone": names[k], "Attention": f"{v['heat_share']:.0%}", "Area": f"{v['area_share']:.0%}",
             "Focus (attention / area)": round(v["focus"], 2)} for k, v in zones.items()]),
            hide_index=True)
        st.caption("Zones are geometric approximations on the 2-D slice, not an anatomical segmentation.")


def show_speech(speech: dict):
    st.markdown(f"**Result: {speech['predicted_class']}** · probability of dementia {speech['p_dementia']:.0%} "
                f"· transcribed with {speech.get('engine', '')}")
    st.markdown("**Transcript** - red words pushed the model towards Dementia, green towards Control")
    st.markdown(f"<div style='background:rgba(128,128,128,0.10);padding:12px;border-radius:4px;line-height:1.9'>"
                f"{highlight_transcript(speech)}</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown("**Speech markers** compared with the training data")
    reading = {"towards_dementia": "⚠️ differs from healthy speech", "towards_control": "✅ healthy",
               "normal": "usual range"}
    st.dataframe(pd.DataFrame([
        {"Marker": m["label"], "This sample": round(m["value"], 2),
         "Control average": round(m.get("control_mean", float("nan")), 2),
         "Dementia average": round(m.get("dementia_mean", float("nan")), 2),
         "Reading": reading[m["flag"]]} for m in speech["markers"]]), hide_index=True)

    st.markdown("**Most influential words** - how much each word raised (+) or lowered (-) the dementia probability")
    top = [w for w in speech["word_contributions"] if abs(w["contribution"]) >= 0.005][:12]
    if top:
        chart = pd.DataFrame({"word": [w["word"] for w in top],
                              "effect on P(dementia)": [w["contribution"] for w in top]}).set_index("word")
        st.bar_chart(chart, horizontal=True, height=320)
    else:
        st.caption("No single word changed the result noticeably.")
    if speech.get("timing"):
        t = speech["timing"]
        a, b, c = st.columns(3)
        a.metric("Speaking rate", f"{t['words_per_minute']:.0f} words/min")
        b.metric("Pauses of 2 s or more", t["long_pauses"])
        c.metric("Recording length", f"{t['duration_seconds']:.0f} s")


def show_guidance(guidance: dict):
    st.info(guidance["headline"])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("🚫 Things to avoid")
        for a in guidance["avoid"]:
            st.markdown(f"**{a['item']}**  \n{a['why']}")
        if guidance["avoid_communication"]:
            st.markdown("##### When talking with the person")
            for a in guidance["avoid_communication"]:
                st.markdown(f"**{a['item']}**  \n{a['why']}")
    with c2:
        st.subheader("✅ Things to do")
        for d in guidance["do"]:
            st.markdown(f"- {d}")
        st.subheader("➡️ Next steps")
        for n in guidance["next_steps"]:
            st.markdown(f"- {n}")
    with st.expander("Risk factors everyone should avoid", expanded=True):
        for a in guidance["avoid_general"]:
            st.markdown(f"- **{a['item']}** - {a['why']}")
    with st.expander("Sources for this guidance"):
        for s in guidance["sources"]:
            st.markdown(f"- {s}")


# --------------------------------------------------------------------- main
def main():
    st.title(" Multimodal Explainable AI for Early Detection of Alzheimer's Disease")
    st.caption("Brain MRI (ResNet50 + Grad-CAM)  +  Speech (speech-to-text + NLP)  →  combined, explained result")
    settings = sidebar()
    available = settings["available"]
    if not any(available.values()):
        st.warning("No trained models found. Follow the steps in README.md to train them, then reload this page.")
        st.stop()

    image, audio, text = inputs(available)
    st.divider()
    ready = image is not None or audio is not None or bool(text and text.strip())
    if st.button("Analyse", type="primary", disabled=not ready):
        w_mri = settings["w_mri"]
        try:
            with st.spinner("Analysing... (the first speech run downloads the speech-to-text model)"):
                st.session_state["result"] = pipeline.analyse(
                    mri_image=image, audio=audio, text=text, whisper_size=settings["whisper"],
                    w_mri=w_mri, w_speech=None if w_mri is None else 1.0 - w_mri)
        except Exception as err:
            st.session_state.pop("result", None)
            st.error(f"Analysis failed: {err}")

    result = st.session_state.get("result")
    if not result:
        st.info("Add an MRI image and/or a speech sample, then press **Analyse**.")
        return

    tabs = st.tabs(["Result and reasons", "MRI explanation", "Speech explanation", "Things to avoid", "Report"])
    with tabs[0]:
        show_overview(result)
    with tabs[1]:
        if result["mri"]:
            show_mri(result["mri"])
        else:
            st.caption("No MRI was provided.")
    with tabs[2]:
        if result["speech"]:
            show_speech(result["speech"])
        else:
            st.caption("No speech sample was provided.")
    with tabs[3]:
        show_guidance(result["guidance"])
    with tabs[4]:
        st.write("Download everything on this page as a single HTML file (opens in any browser, prints to PDF).")
        st.download_button("Download report", html_report(result), file_name="alzheimers_analysis_report.html",
                           mime="text/html")
    st.caption(config.DISCLAIMER)


main()
