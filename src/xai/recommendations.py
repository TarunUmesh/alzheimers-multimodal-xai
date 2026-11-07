"""Stage-wise guidance: what to avoid, what to do, and what to do next.

This is a fixed knowledge base, selected by the predicted stage and by the
speech findings. It is general health information drawn from the public
sources listed in SOURCES below - it is not personal medical advice.
"""
from __future__ import annotations

SOURCES = [
    "World Health Organization (2019). Risk reduction of cognitive decline and dementia: WHO guidelines.",
    "Livingston G. et al. (2024). Dementia prevention, intervention, and care: 2024 report of the "
    "Lancet standing Commission. The Lancet, 404(10452), 572-628.",
    "Alzheimer's Association. Stages of Alzheimer's and caregiving guidance (alz.org).",
    "National Institute on Aging. Alzheimer's caregiving and home-safety guidance (nia.nih.gov).",
]

# Applies to everyone - these are the modifiable risk factors.
GENERAL_AVOID = [
    ("Smoking and tobacco in any form", "Smoking damages blood vessels in the brain; stopping lowers risk even late in life."),
    ("Heavy or regular alcohol drinking", "Excess alcohol is directly toxic to brain cells and raises dementia risk."),
    ("Sitting for most of the day", "Physical inactivity is one of the largest modifiable risk factors."),
    ("Leaving high blood pressure, diabetes or high cholesterol untreated", "These damage the small blood vessels that feed the brain."),
    ("Withdrawing from friends, family and community", "Social isolation speeds up cognitive decline."),
    ("Ignoring hearing or eyesight problems", "Untreated hearing and vision loss are linked to faster decline; hearing aids and glasses help."),
    ("Head injuries", "Wear a helmet and seat belt, and remove trip hazards at home."),
    ("Irregular or very short sleep", "Poor sleep worsens memory and attention."),
    ("Taking 'memory' supplements or changing medicines without a doctor", "Vitamin B, vitamin E and omega-3 supplements are not recommended by WHO for reducing dementia risk, and some sleeping or allergy tablets worsen confusion."),
]

STAGE_GUIDANCE = {
    "NonDemented": {
        "headline": "No sign of dementia was detected in what was analysed.",
        "avoid": [
            ("Treating this result as a medical all-clear", "If memory or language problems are noticed in daily life, see a doctor regardless of this output."),
        ],
        "do": [
            "Keep physically active - about 150 minutes of moderate activity a week.",
            "Keep the mind engaged: reading, learning, puzzles, music, a new skill.",
            "Eat a balanced, Mediterranean-style diet with plenty of vegetables, fruit, pulses and fish.",
            "Have blood pressure, blood sugar and cholesterol checked regularly.",
        ],
        "next": ["No action is needed on the basis of this result. Repeat a check-up if symptoms appear."],
    },
    "VeryMildDemented": {
        "headline": "The pattern is consistent with very mild (early) changes.",
        "avoid": [
            ("Dismissing forgetfulness as 'just old age'", "Early assessment gives the widest choice of treatment and planning."),
            ("Giving up hobbies, work or social activities", "Staying engaged helps to maintain abilities for longer."),
            ("Relying on memory alone for medicines, bills and appointments", "Use a pill organiser, calendar and phone reminders."),
            ("Doing several demanding things at once", "One task at a time reduces mistakes and frustration."),
            ("Postponing legal and financial planning", "It is best arranged while the person can fully take part in the decisions."),
        ],
        "do": [
            "Book an appointment with a neurologist or memory clinic for a proper assessment.",
            "Keep a daily routine and a written notebook or phone notes.",
            "Stay physically and socially active.",
            "Tell a trusted family member what is happening.",
        ],
        "next": [
            "See a doctor for cognitive testing (for example MMSE or MoCA).",
            "Ask about blood tests for treatable causes of memory problems: vitamin B12, thyroid, depression, medicine side-effects.",
        ],
    },
    "MildDemented": {
        "headline": "The pattern is consistent with mild dementia.",
        "avoid": [
            ("Driving until a doctor has assessed fitness to drive", "Judgement and reaction time can be affected before the person notices."),
            ("Managing medicines and money entirely alone", "Missed or double doses and financial mistakes become common; share the task."),
            ("Sudden changes to routine or surroundings", "Familiar routine reduces confusion and anxiety."),
            ("Arguing, correcting or testing ('Don't you remember?')", "It causes distress and does not improve memory."),
            ("Clutter, loose rugs and poor lighting at home", "Falls are a major risk."),
            ("Stopping prescribed medicines without medical advice", "Changes should be made by the treating doctor."),
        ],
        "do": [
            "Arrange a specialist assessment soon if one has not been done.",
            "Label cupboards and keep important items in fixed places.",
            "Use short, clear sentences and allow extra time for answers.",
            "Keep up walking, light exercise and familiar social contact.",
        ],
        "next": [
            "Consult a neurologist or geriatrician; ask about treatment options and support services.",
            "Carry identification with an emergency contact number.",
        ],
    },
    "ModerateDemented": {
        "headline": "The pattern is consistent with moderate dementia.",
        "avoid": [
            ("Leaving the person alone for long periods", "Wandering, falls and accidents are much more likely at this stage."),
            ("Driving", "It is no longer safe."),
            ("Unsupervised use of the stove, medicines, sharp tools or cleaning chemicals", "Keep them locked or supervised."),
            ("Noisy, crowded or unfamiliar places", "They can cause agitation and distress."),
            ("Arguing, rushing, or contradicting what the person believes", "Reassure and gently redirect instead."),
            ("Carers trying to cope alone", "Carer exhaustion is common and harms both people; accept help and take breaks."),
        ],
        "do": [
            "Ensure regular medical follow-up and review of medicines.",
            "Keep a calm, fixed daily routine with help for dressing, eating and washing.",
            "Fit door alarms or locks if wandering occurs; keep an ID bracelet on the person.",
            "Use music, photographs and familiar activities for comfort.",
        ],
        "next": [
            "See the treating doctor promptly to plan care and support.",
            "Contact a local dementia support organisation for carer help.",
        ],
    },
}

COMMUNICATION_AVOID = [
    ("Finishing the person's sentences or interrupting", "Word-finding takes longer; give time."),
    ("Open-ended questions such as 'What do you want to eat?'", "Offer two choices instead: 'Tea or coffee?'"),
    ("Speaking in a noisy room or from behind", "Face the person and reduce background noise."),
]

CONFLICT_AVOID = ("Drawing a conclusion from one result when the two disagree",
                  "The scan and the speech sample point in different directions; only a clinical assessment can resolve this.")


def stage_from_results(mri: dict | None, fusion: dict) -> str:
    """The MRI stage decides the guidance; without an MRI the fused risk level is used."""
    if mri is not None:
        stage = mri["predicted_class"]
        # Speech clearly abnormal but MRI normal -> give the cautious early-stage guidance.
        if stage == "NonDemented" and fusion["risk_level"] != "Low":
            return "VeryMildDemented"
        return stage
    return {"Low": "NonDemented", "Moderate": "VeryMildDemented", "High": "MildDemented"}[fusion["risk_level"]]


def build_guidance(mri: dict | None, speech: dict | None, fusion: dict) -> dict:
    stage = stage_from_results(mri, fusion)
    base = STAGE_GUIDANCE[stage]
    avoid = list(base["avoid"])

    if fusion["agreement"] == "conflict":
        avoid.insert(0, CONFLICT_AVOID)

    speech_flags = {m["name"] for m in (speech or {}).get("markers", []) if m["flag"] == "towards_dementia"}
    word_finding = speech_flags & {"filler_rate", "vague_rate", "uncertainty_rate", "repetition_rate"}
    communication = COMMUNICATION_AVOID if (speech and speech["predicted_class"] == "Dementia" and word_finding) else []

    return {
        "guidance_stage": stage,
        "headline": base["headline"],
        "avoid": [{"item": a, "why": w} for a, w in avoid],
        "avoid_communication": [{"item": a, "why": w} for a, w in communication],
        "avoid_general": [{"item": a, "why": w} for a, w in GENERAL_AVOID],
        "do": list(base["do"]),
        "next_steps": list(base["next"]),
        "sources": SOURCES,
    }
