"""PulseIQ — cardiometabolic risk screening from fingertip photoplethysmography.

The UI follows bedside patient-monitor convention: the plethysmograph trace is
cyan (the pulse-oximetry channel), derived heart rate is green, and findings are
graded with IEC 60601-1-8 alarm colours (red = immediate, yellow = prompt).
All colour comes from .streamlit/config.toml — no injected CSS.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer


from ppg import (
    ANALYSIS_FS,
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    MIN_WINDOW_S,
    WINDOW_S,
    PulseSampler,
    PulseTracker,
    estimate_pulse,
    samples_from_video,
)

WHO_PREVALENCE_REFERENCE = {
    "diabetes": {
        "adults_18plus_overall_pct": 14,
        "adults_65plus_pct": 24,
        "sex_note": "algo más alta en hombres que en mujeres",
        "age_note": "aumenta con la edad; pico alrededor de los 75-79 años",
        "source": "OMS / IDF Diabetes Atlas 2024",
    },
    "hypertension": {
        "adults_30_79_overall_pct": 33,
        "awareness_note": "cerca del 46% no sabe que la tiene",
        "sex_note": "algo más frecuente en hombres",
        "age_note": "aumenta marcadamente con la edad",
        "source": "OMS, hoja informativa hipertensión",
    },
}
# ============================== Configuration ==============================

API_URL = os.environ.get(
    "API_URL", "https://pulseiq-api-431111687933.europe-west1.run.app/predict"
)

# Monitor channel colours. These mirror the semantic colours in config.toml so
# that Altair traces match the :green[...] / :blue[...] markdown readouts.
C_PLETH = "#38BDF8"  # plethysmograph — the pulse-ox channel
C_HR = "#4ADE80"  # heart rate
C_GRID = "#22304A"
C_MUTED = "#94A3B8"

# Resting adult heart rate, per AHA. Outside this band we raise an advisory.
HR_REFERENCE = (60, 100)

# WHO adult BMI categories.
BMI_BANDS = [
    (18.5, "Underweight", "blue"),
    (25.0, "Normal", "green"),
    (30.0, "Overweight", "yellow"),
    (float("inf"), "Obese", "red"),
]

st.set_page_config(
    page_title="PulseIQ — cardiometabolic screening",
    page_icon=":material/cardiology:",
    layout="wide",
    # The device panel (reference ranges, channel legend, regulatory notice) is
    # part of the instrument, not an optional extra — show it by default.
    initial_sidebar_state="expanded",
)


def gemini_api_key() -> str:
    """Read the Gemini key from secrets, falling back to the environment.

    Accessing st.secrets raises when no secrets.toml exists, so this must stay
    defensive — the app should still run (and screen) without a key.
    """
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return (key or os.environ.get("GEMINI_API_KEY", "")).strip()


# ============================== Session state ==============================

DEFAULTS = {
    "step": 1,
    "patient_ref": "",
    "age": 25,
    "gender": "Male",
    "height": 1.70,
    "weight": 68.90,
    "ever_smoked": "No",
    "current_smoker": "No",
    "measured_bpm": None,
    "signal_pct": None,
    "source": None,
    "api_result": None,
    "recommendation": None,
    "analysis": None,  # what was analysed for the accepted reading (summary + report)
    "live_was_playing": False,
    "_pulse_was_locked": False,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

# Per-session acquisition objects, created once. The sampler is handed to the
# WebRTC worker thread as a bound method, so it must be owned by this session —
# a module-level buffer would be shared (and clobbered) across users.
if "pulse_sampler" not in st.session_state:
    st.session_state.pulse_sampler = PulseSampler()
if "pulse_tracker" not in st.session_state:
    st.session_state.pulse_tracker = PulseTracker()
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8].upper()


def grade_quality(pct: float):
    """Map a signal-quality percentage to (label, colour, acceptable).

    TODO(clinical): these cut-offs decide when a reading is trustworthy enough
    to screen on. Tune them against your own recordings — see the note in the
    handover. `acceptable` gates the "proceed to assessment" button.
    """
    if pct >= 55.0:
        return "Good", "green", True
    if pct >= 30.0:
        return "Fair", "yellow", True
    return "Poor", "red", False


def classify_bmi(bmi: float):
    """Return the (label, colour) WHO band for an adult BMI."""
    for upper, label, colour in BMI_BANDS:
        if bmi < upper:
            return label, colour
    return "Obese", "red"


def grade_hr(bpm: float):
    """Return (label, colour) for a resting heart rate against the AHA range."""
    low, high = HR_REFERENCE
    if bpm < low:
        return "Bradycardic", "yellow"
    if bpm > high:
        return "Tachycardic", "yellow"
    return "In range", "green"


def bsa_dubois(height_m: float, weight_kg: float) -> float:
    """Body surface area by the Du Bois formula, in m².

    BSA is the standard denominator for dosing and cardiac index, so it earns
    its place next to BMI on a clinical intake panel.
    """
    if height_m <= 0 or weight_kg <= 0:
        return 0.0
    return 0.007184 * (height_m * 100) ** 0.725 * weight_kg**0.425


@st.cache_data(show_spinner=False, ttl="30m", max_entries=8)
def extract_samples_from_video(video_bytes, max_seconds=20.0):
    """Cached decode of an uploaded recording into FrameSamples.

    Up to 20 s are read; the estimator then analyses the most recent
    WINDOW_S of that, which skips the exposure transient at the start.
    """
    return samples_from_video(video_bytes, max_seconds=max_seconds)


# ============================== Chart helpers ==============================


def pleth_chart(values, height=190):
    """Monitor-style plethysmograph trace: cyan, axis-free, full bleed.

    Real monitors draw the pleth without axes — the shape and rate carry the
    information, not absolute amplitude (which is uncalibrated anyway).
    """
    frame = pd.DataFrame({"sample": np.arange(len(values)), "amplitude": values})
    return (
        alt.Chart(frame)
        .mark_line(color=C_PLETH, strokeWidth=1.6, interpolate="basis")
        .encode(
            # nice=False: the trace must fill the panel edge to edge, not stop
            # at 90% because the sample count was rounded up to a round number.
            x=alt.X("sample:Q", axis=None, scale=alt.Scale(nice=False)),
            y=alt.Y("amplitude:Q", axis=None, scale=alt.Scale(zero=False)),
        )
        # No Altair title: it renders as visible text inside the SVG, which both
        # duplicates the st.caption above the chart and puts a heading over a
        # deliberately axis-free trace. The caption is real DOM text, so screen
        # readers already get the description.
        .properties(height=height)
    )


def spectrum_chart(freqs_bpm, spectrum, peak_bpm, height=190):
    """Power spectrum with the detected pulse marked — the analysis view."""
    frame = pd.DataFrame({"Rate (bpm)": freqs_bpm, "Power": spectrum})
    area = (
        alt.Chart(frame)
        .mark_area(
            line={"color": C_PLETH, "strokeWidth": 1.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#0B1220", offset=0),
                    alt.GradientStop(color=C_PLETH, offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
            opacity=0.35,
        )
        .encode(
            x=alt.X("Rate (bpm):Q", scale=alt.Scale(nice=False)),
            y=alt.Y("Power:Q", axis=None),
            tooltip=["Rate (bpm):Q", "Power:Q"],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"Rate (bpm)": [peak_bpm]}))
        .mark_rule(color=C_HR, strokeWidth=1.5, strokeDash=[4, 3])
        .encode(x="Rate (bpm):Q")
    )
    return (area + marker).properties(height=height)


def readout(label, value, unit, colour="green", note=None):
    """One monitor panel: small channel label, large coloured value, footnote.

    Unit lives in the label rather than beside the number — Streamlit sizes a
    whole markdown line at once, so an inline unit would render at the same
    46px as the value. Monitors label the channel and unit small anyway, and
    give the whole visual budget to the digits.
    """
    st.caption(f"{label} · {unit}" if unit else label)
    st.markdown(f"# :{colour}[{value}]")
    if note:
        st.caption(note)


def risk_bar(probability: float, colour: str, height=46):
    """Probability bar with the 50% decision threshold drawn on it.

    A bare progress bar shows magnitude but not the thing that actually decides
    the finding — where the value sits relative to the threshold. The dashed
    rule makes "2%" and "48%" read differently at a glance.
    """
    track = (
        alt.Chart(pd.DataFrame({"v": [100.0]}))
        .mark_bar(color="#1B2740", cornerRadius=2)
        .encode(x=alt.X("v:Q", scale=alt.Scale(domain=[0, 100]), axis=None))
    )
    fill = (
        alt.Chart(pd.DataFrame({"v": [probability]}))
        .mark_bar(color=colour, cornerRadius=2)
        .encode(x=alt.X("v:Q", scale=alt.Scale(domain=[0, 100]), axis=None))
    )
    threshold = (
        alt.Chart(pd.DataFrame({"v": [50.0]}))
        .mark_rule(color="#94A3B8", strokeWidth=1.5, strokeDash=[3, 3])
        .encode(x=alt.X("v:Q", scale=alt.Scale(domain=[0, 100])))
    )
    return (track + fill + threshold).properties(height=height)


def build_report(
    *, bmi, bsa, pulse, hr_label, quality, source, diabetic, hypertensive, analysis=None
):
    """Render the session as a plain-text clinical summary for download."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    acquisition_detail = ""
    if analysis:
        ibi = "n/a" if analysis["ibi_bpm"] is None else f"{analysis['ibi_bpm']:.1f} bpm"
        acquisition_detail = (
            f"  Window           {analysis['duration_s']:.1f} s at {analysis['fps']:.0f} fps, "
            f"{analysis['channel']} channel\n"
            f"  Spectral rate    {analysis['spectral_bpm']:.1f} bpm\n"
            f"  Beat intervals   {ibi} over {analysis['n_beats']} beats "
            f"({analysis['beat_consistency'] * 100:.0f}% regular)\n"
            f"  Captured         {analysis['captured_at']} ({analysis['status']})\n"
        )
    return f"""PULSEIQ SCREENING SUMMARY
========================================
Session      {st.session_state.session_id}
Patient ref  {st.session_state.patient_ref or "(not supplied)"}
Generated    {stamp}

PATIENT
  Age              {st.session_state.age} years
  Sex              {st.session_state.gender}
  Height           {st.session_state.height:.2f} m
  Weight           {st.session_state.weight:.1f} kg
  BMI              {bmi:.1f} kg/m² ({classify_bmi(bmi)[0]}, WHO band)
  BSA (Du Bois)    {bsa:.2f} m²
  Ever smoked      {st.session_state.ever_smoked}
  Current smoker   {st.session_state.current_smoker}

ACQUISITION
  Modality         Fingertip PPG ({source})
  Passband         {BAND_LOW_HZ}-{BAND_HIGH_HZ} Hz
  Pulse rate       {pulse} bpm ({hr_label}; AHA ref {HR_REFERENCE[0]}-{HR_REFERENCE[1]})
  Signal quality   {quality:.0f}%
{acquisition_detail}
RISK MODEL OUTPUT (threshold 50%)
  Diabetes         {diabetic:.1f}%  {"ELEVATED" if diabetic >= 50 else "within expected range"}
  Hypertension     {hypertensive:.1f}%  {"ELEVATED" if hypertensive >= 50 else "within expected range"}

----------------------------------------
Investigational software. Not a medical device and not for diagnostic
use. Screening output does not constitute a diagnosis. Confirm any
finding by standard clinical measurement.
"""


# ============================== Recommendation ==============================


def get_gemini_recommendation(diabetic, hypertensive, age, bmi, pulse, sex):
    """Ask Gemini for a structured lifestyle recommendation.

    Returns a dict with keys: diet, exercise, monitoring (each a list of
    strings), further_reading (list of {title, url}) and disclaimer. On error
    returns a dict with an 'error' key instead.
    """
    api_key = gemini_api_key()
    if not api_key:
        return {
            "error": "No GEMINI_API_KEY found. Add it to "
            "frontend/.streamlit/secrets.toml locally, or to Streamlit Secrets "
            "when deployed."
        }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash:generateContent"
    )
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    diabetes_status = "Elevated risk" if diabetic == 1 else "Standard risk"
    ht_status = "Elevated risk" if hypertensive == 1 else "Standard risk"

    system_instruction = """You are an empathetic, knowledgeable health and lifestyle educator.
    Return lifestyle recommendations as JSON ONLY — no markdown, no code fences, no text outside the JSON object.

    The JSON must have exactly this shape:
    {
    "population_context": ["detailed statement 1", "detailed statement 2"],
    "diet": ["detailed tip 1", "detailed tip 2"],
    "exercise": ["detailed tip 1", "detailed tip 2"],
    "monitoring": ["detailed tip 1", "detailed tip 2"],
    "further_reading": [{"title": "WHO – Healthy diet", "url": "https://www.who.int/..."}],
    "disclaimer": "one-line disclaimer text"
    }

    RULES:
    1. "population_context" must have 2 statements written in a WARM, PERSONAL, second-person voice
    (matching the rest of the guidance). Do NOT write an impersonal fact sheet.
    Start by acknowledging that you can see THIS person's values, then place them relative to
    their peers. Model the tone on this example:
    "I can see your values, and for people of your age and sex, you fall within the expected
    range. In your age group, diabetes prevalence sits at roughly 1% to 15%, a figure that rises
    with age, according to the WHO and the IDF Diabetes Atlas 2024."
    - Weave in the person's OWN data from the profile — their age, sex, BMI, resting pulse — so it
        reads as tailored to them specifically.
    - Give the prevalence for THEIR age group as an APPROXIMATE RANGE, derived from the WHO/IDF
        reference data provided, and state that their values fall WITHIN the expected range for their
        group when that is the case.
    - This is a COMPARISON to their group, NOT a personal prediction. NEVER write "you have an
        X% chance of developing..."; always frame it as "you fall within the expected range for
        your group". Keep it clearly separate from the ML risk assessment.
    2. Each list should have 2-4 detailed, verbose tips that explain the reasoning, not just the instruction.
    3. Where relevant, include concrete statistics (e.g. 150 minutes/week of moderate activity, healthy BMI range 18.5-24.9) and briefly attribute the source type (WHO, CDC, ADA, NHS).
    4. "further_reading" must contain 3-5 links using only these real domains: who.int, cdc.gov, diabetes.org, nhs.uk, mayoclinic.org. Prefer main topic pages; do not invent deep URLs.
    5. NEVER provide a medical diagnosis or diagnostic language. Population figures must stay descriptive, framing the person as within (or outside) the expected range for their group — never predicting this individual's future.
    6. The "disclaimer" value must be exactly: "This is an automated lifestyle suggestion based on statistical data, not medical advice. Always consult a healthcare professional."
    7. Output valid JSON only.
    """

    user_prompt = f"""Patient Profile:
- Age: {age}
- Sex: {sex}
- BMI: {bmi}
- Resting Pulse: {pulse} bpm

Machine Learning Risk Assessment:
- Diabetes: {diabetes_status}
- Hypertension: {ht_status}

WHO / IDF reference prevalence data (global adults) — use ONLY these figures for
population_context, do not recall any others from memory:
{json.dumps(WHO_PREVALENCE_REFERENCE, ensure_ascii=False)}

Generate the recommendations as JSON."""
    body = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 1500,
            "temperature": 0.3,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            text = (
                text.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            return json.loads(text)
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            if status in (503, 429) and attempt < 2:
                time.sleep(2)
                continue
            try:
                error_details = (
                    e.response.json()
                    .get("error", {})
                    .get("message", "Unknown API error")
                )
            except Exception:
                error_details = "Could not parse API error response."
            return {"error": f"API error {status}: {error_details}"}
        except (KeyError, IndexError):
            return {"error": "Gemini returned an unexpected response format."}
        except json.JSONDecodeError:
            return {"error": "Gemini did not return valid JSON. Try again."}


def store_reading(record):
    """Accept an analysed window as the session's pulse reading."""
    st.session_state.measured_bpm = record["bpm"]
    st.session_state.signal_pct = record["quality"]
    st.session_state.source = record["source"]
    st.session_state.analysis = record


def clear_reading():
    st.session_state.measured_bpm = None
    st.session_state.signal_pct = None
    st.session_state.source = None
    st.session_state.analysis = None


def reset_acquisition():
    """Forget buffered frames, tracker history and the reading: start clean."""
    st.session_state.pulse_sampler.clear()
    st.session_state.pulse_tracker.reset()
    st.session_state._pulse_was_locked = False
    st.session_state.live_was_playing = False
    clear_reading()


def goto(step: int):
    """Advance the workflow and clear anything downstream of the new step."""
    if step <= 2:
        st.session_state.api_result = None
        st.session_state.recommendation = None
    st.session_state.step = step
    st.rerun()


# ============================== Instrument header ==============================

step = st.session_state.step

STEPS = [
    ("Patient", ":material/badge:"),
    ("Acquisition", ":material/sensors:"),
    ("Assessment", ":material/monitor_heart:"),
]

# Three bordered cards cost ~90px of vertical rhythm to say "you are on step 1".
# One badge row says the same thing in a single line and keeps the fold for the
# waveform, which is the content people actually came for.
marks = []
for index, (name, _) in enumerate(STEPS, start=1):
    if index < step:
        marks.append(f":green-badge[:material/check: {name}]")
    elif index == step:
        marks.append(f":blue-badge[{index}. {name}]")
    else:
        marks.append(f":gray-badge[{index}. {name}]")

with st.container(horizontal=True, vertical_alignment="center"):
    st.header(":material/cardiology: PulseIQ", width="content")
    st.markdown(" ".join(marks), width="content")
    # Each item is its own element so a narrow viewport wraps *between* them.
    # Joining into one string let the line break mid-timestamp ("2026-" / "09-02").
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.session_state.patient_ref:
            st.markdown(f":gray[Ref] `{st.session_state.patient_ref}`", width="content")
        st.markdown(f":gray[Session] `{st.session_state.session_id}`", width="content")
        st.markdown(
            f":gray[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC]",
            width="content",
        )

# ============================== Sidebar: device panel ==============================

with st.sidebar:
    st.markdown("**Device**")
    st.table(
        {
            "Modality": "Fingertip PPG",
            "Passband": f"{BAND_LOW_HZ}–{BAND_HIGH_HZ} Hz",
            "Window": f"{MIN_WINDOW_S:.0f}–{WINDOW_S:.0f} s · {ANALYSIS_FS:.0f} Hz",
            "Inference": "Remote (XGBoost)",
        },
        border="horizontal",
        width="stretch",
    )

    st.markdown("**Reference ranges**")
    st.table(
        {
            "Resting HR": f"{HR_REFERENCE[0]}–{HR_REFERENCE[1]} bpm · AHA",
            "BMI normal": "18.5–24.9 kg/m² · WHO",
            "Risk threshold": "≥ 50% flags elevated",
        },
        border="horizontal",
        width="stretch",
    )

    st.markdown("**Channel legend**")
    # Badges rather than a coloured glyph: U+25AC is absent from IBM Plex and
    # falls back to an uncoloured dash.
    st.markdown(
        ":blue-badge[PLETH] Plethysmograph  \n"
        ":green-badge[HR] Heart rate  \n"
        ":yellow-badge[MED] Prompt response  \n"
        ":red-badge[HIGH] Immediate response"
    )
    st.caption("Colours follow IEC 60601-1-8 and bedside-monitor parameter convention.")

    # Standing regulatory notice. Kept subtle here because step 1 shows the
    # prominent version at entry — two identical yellow boxes on one screen
    # read as a rendering bug, not as emphasis.
    st.caption(
        ":material/gpp_maybe: Investigational software · not a medical device · "
        "not for diagnostic use"
    )

# ==========================================================================
# STEP 1 — PATIENT
# ==========================================================================
if step == 1:
    # Informed consent belongs at the entry point, but three lines of yellow
    # above the fold buries the form it introduces. One line, with the detail
    # a hover away.
    st.warning(
        "Screening estimates only, not a diagnosis. Confirm any finding with "
        "standard clinical measurement.",
        icon=":material/gpp_maybe:",
        title="Investigational software — not a medical device",
    )

    st.subheader("Patient record")

    # Inputs on the left, derived values in a narrow right rail. The old
    # full-width BMI card spent ~60% of its area on empty background.
    intake, derived = st.columns([2.2, 1])

    with intake:
        # One panel with aligned rows, not three sub-panels of unequal height —
        # side-by-side cards left ~130px of dead space under the shorter one,
        # and three sets of borders for seven fields is border noise.
        with st.container(border=True, height="stretch"):
            ref_col, age_col, sex_col = st.columns([2, 1, 2])
            with ref_col:
                patient_ref = st.text_input(
                    "Patient name",
                    value=st.session_state.patient_ref,
                    max_chars=24,
                    placeholder="your name",
                    icon=":material/badge:",
                    help="Optional. Enter a medical record number (MRN) or study ID for tracking. Max 24 characters.",
                )
            with age_col:
                age = st.number_input(
                    "Age",
                    min_value=0,
                    max_value=120,
                    value=st.session_state.age,
                )
            with sex_col:
                gender = st.segmented_control(
                    "Sex",
                    ["Male", "Female"],
                    default=st.session_state.gender,
                    width="stretch",
                )

            height_col, weight_col = st.columns(2)
            with height_col:
                height = st.number_input(
                    "Height (m)",
                    min_value=1.0,
                    max_value=2.5,
                    value=st.session_state.height,
                    step=0.01,
                    format="%.2f",
                    help="Height in metres (1.0–2.5 m). Used to calculate BMI and body surface area (BSA).",
                )
            with weight_col:
                weight = st.number_input(
                    "Weight (kg)",
                    min_value=20.0,
                    max_value=250.0,
                    value=st.session_state.weight,
                    # step must divide (value - min_value) or the browser marks
                    # the input aria-invalid. With step=0.5 from min=20.0 only
                    # .0/.5 are valid, so the 68.9 default rendered as invalid.
                    step=0.1,
                    format="%.1f",
                    help="Weight in kilograms (20–250 kg). Used to calculate BMI and body surface area (BSA).",
                )

            smoked_col, smoker_col = st.columns(2)
            with smoked_col:
                ever_smoked = st.segmented_control(
                    "Previous smoker",
                    ["No", "Yes"],
                    default=st.session_state.ever_smoked,
                    width="stretch",
                )
            with smoker_col:
                current_smoker = st.segmented_control(
                    "Current smoker",
                    ["No", "Yes"],
                    default=st.session_state.current_smoker,
                    width="stretch",
                )

    # segmented_control returns None if the user deselects; hold the last value.
    gender = gender or st.session_state.gender
    ever_smoked = ever_smoked or st.session_state.ever_smoked
    current_smoker = current_smoker or st.session_state.current_smoker

    bmi = round(weight / (height**2), 2) if height > 0 else 0.0
    bmi_label, bmi_colour = classify_bmi(bmi)
    bsa = bsa_dubois(height, weight)

    # Inline validation feedback for critical measurements
    validation_issues = []
    if height < 1.2 or height > 2.4:
        validation_issues.append(
            f"Height {height:.2f}m is outside typical adult range (1.2–2.4m)"
        )
    if weight < 30 or weight > 200:
        validation_issues.append(
            f"Weight {weight:.1f}kg is outside typical adult range (30–200kg)"
        )
    if age < 18:
        validation_issues.append(
            "Age is below 18 years — this screening is designed for adults"
        )

    if validation_issues:
        for issue in validation_issues:
            st.info(f"⚠️ {issue}", icon=":material/info:")
        st.markdown("_Please verify these values are correct before proceeding._")

    with derived:
        with st.container(border=True, height="stretch"):
            st.markdown("**Derived**")
            readout(
                "BMI · body mass index",
                f"{bmi:.1f}",
                "kg/m²",
                colour=bmi_colour,
                note=f"{bmi_label} · WHO adult band",
            )
            st.markdown(f":gray[BSA · Du Bois] **{bsa:.2f}** :gray[m²]")

    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button(
            "Begin acquisition",
            type="primary",
            icon=":material/arrow_forward:",
            shortcut="Mod+Enter",
            help="Keyboard shortcut: Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux)",
        ):
            st.session_state.patient_ref = patient_ref.strip()
            st.session_state.age = age
            st.session_state.gender = gender
            st.session_state.height = height
            st.session_state.weight = weight
            st.session_state.ever_smoked = ever_smoked
            st.session_state.current_smoker = current_smoker
            reset_acquisition()
            goto(2)

# ==========================================================================
# STEP 2 — ACQUISITION
# ==========================================================================
elif step == 2:
    st.subheader("Pulse acquisition")

    mode = st.segmented_control(
        "Acquisition mode",
        ["Upload recording", "Live camera"],
        default="Upload recording",
        width="stretch",
        help="Upload: pre-recorded video of fingertip PPG. Live: real-time camera capture (requires device with camera).",
    )

    CHANNEL_NAMES = {"r": "Red", "g": "Green", "b": "Blue"}
    STATUS_BADGES = {
        "acquiring": ("Acquiring", "blue"),
        "stable": ("Steady", "blue"),
        "holding": ("Holding", "yellow"),
        "searching": ("Searching", "gray"),
    }
    QUALITY_HELP = (
        "Share of in-band spectral power at the pulse fundamental and its second "
        "harmonic, scaled by beat-to-beat regularity. A clean PPG concentrates "
        "energy at the pulse frequency with evenly spaced beats; motion artefact "
        "spreads it across the band."
    )

    def render_trace(estimate, source, bpm, quality, status=None):
        """Draw the monitor for one analysis window: pleth, HR, quality, spectrum.

        `bpm` and `quality` are what the readout shows (for live mode the
        tracker's smoothed value, or None while no window has passed the
        quality gate); `estimate` supplies the waveform and spectrum of the
        current window. Returns whether `quality` is acceptable for screening.
        """
        q_label, q_colour, acceptable = grade_quality(quality)
        trace_col, vitals_col = st.columns([3, 1], border=True)

        with trace_col:
            st.caption(
                f"PLETH · {CHANNEL_NAMES[estimate.channel]} channel, drift removed · "
                f"{estimate.duration_s:.1f} s window · camera {estimate.fps:.0f} fps · "
                f"resampled to {ANALYSIS_FS:.0f} Hz"
            )
            with st.skeleton(height=200):
                st.altair_chart(pleth_chart(estimate.filtered))

        with vitals_col:
            if bpm is None:
                # No window has passed the quality gate yet: a monitor blanks
                # the number rather than show a guess.
                readout("HR · heart rate", "- -", "bpm", colour="gray")
                badges = ":gray-badge[No lock]"
            else:
                hr_label, hr_colour = grade_hr(bpm)
                readout("HR · heart rate", f"{bpm:.0f}", "bpm", colour=hr_colour)
                badges = f":{hr_colour}-badge[{hr_label}]"
            if status:
                badges += f" :{status[1]}-badge[{status[0]}]"
            st.markdown(badges)
            st.markdown(
                f":gray[Signal] :{q_colour}[{quality:.0f}%] :gray[· {q_label}]",
                help=QUALITY_HELP,
            )
            st.progress(min(max(quality / 100.0, 0.0), 1.0))
            st.caption(f"Source · {source}")

        with st.container(border=True):
            st.caption(
                f"SPECTRUM · in-band power · dashed line marks the spectral peak "
                f"at {estimate.spectral_bpm:.0f} bpm"
            )
            with st.skeleton(height=200):
                st.altair_chart(
                    spectrum_chart(
                        estimate.freqs_bpm, estimate.spectrum, estimate.spectral_bpm
                    )
                )
        return acceptable

    def analysis_record(estimate, bpm, quality, source, status):
        """What was analysed, kept in session state for the summary and report."""
        return {
            "bpm": float(bpm),
            "quality": float(quality),
            "source": source,
            "status": status,
            "duration_s": float(estimate.duration_s),
            "fps": float(estimate.fps),
            "channel": CHANNEL_NAMES[estimate.channel],
            "spectral_bpm": float(estimate.spectral_bpm),
            "ibi_bpm": None if estimate.ibi_bpm is None else float(estimate.ibi_bpm),
            "n_beats": int(estimate.n_beats),
            "beat_consistency": float(estimate.beat_consistency),
            "captured_at": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        }

    def render_analysis(record):
        """The evidence behind the number: what was analysed, and whether the
        two independent rate estimates (spectrum, beat intervals) agree."""
        q_label, q_colour, _ = grade_quality(record["quality"])
        hr_label, hr_colour = grade_hr(record["bpm"])
        if record["ibi_bpm"] is None:
            ibi = f"— ({record['n_beats']} beats, too few to time)"
        else:
            ibi = f"{record['ibi_bpm']:.1f} bpm over {record['n_beats']} beats"
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("**Analysis summary**", width="content")
                st.markdown(
                    f":gray[{record['source']} · {record['status']} · "
                    f"captured {record['captured_at']}]",
                    width="content",
                )
            st.table(
                {
                    "Pulse rate": f"{record['bpm']:.0f} bpm · :{hr_colour}-badge[{hr_label}]",
                    "Signal quality": f"{record['quality']:.0f}% · :{q_colour}-badge[{q_label}]",
                    "Window analysed": (
                        f"{record['duration_s']:.1f} s at {record['fps']:.0f} fps · "
                        f"{record['channel']} channel"
                    ),
                    "Spectral rate": f"{record['spectral_bpm']:.1f} bpm",
                    "Beat-interval rate": ibi,
                    "Beat regularity": f"{record['beat_consistency'] * 100:.0f}%",
                },
                border="horizontal",
                width="stretch",
            )

    def render_verdict(record, accepted):
        if accepted:
            st.success(
                f"Reading accepted — {record['bpm']:.0f} bpm at "
                f"{record['quality']:.0f}% signal quality. Run the assessment when ready.",
                icon=":material/check_circle:",
            )
        else:
            st.warning(
                "Signal quality is below the acceptance threshold. Re-record with "
                "the fingertip held still and fully covering the lens.",
                icon=":material/warning:",
            )

    def render_idle(message):
        """Monitor waiting state: flat baseline and dashed readouts.

        A real monitor holds its panel layout and shows a flat trace when no
        signal is present. Rendering nothing until a file arrives made the page
        jump by ~400px the moment one did.
        """
        trace_col, vitals_col = st.columns([3, 1], border=True)
        with trace_col:
            st.caption("PLETH · no signal")
            flat = pd.DataFrame({"sample": np.arange(120), "amplitude": np.zeros(120)})
            st.altair_chart(
                alt.Chart(flat)
                .mark_line(color="#2A3A55", strokeWidth=1.6)
                .encode(
                    x=alt.X("sample:Q", axis=None),
                    y=alt.Y("amplitude:Q", axis=None, scale=alt.Scale(domain=[-1, 1])),
                )
                .properties(height=190)
            )
        with vitals_col:
            readout("HR · heart rate", "- -", "bpm", colour="gray")
            st.badge("Standby", color="gray")
            st.caption(message)

    def render_captured():
        """Summary of the reading on file, if any — the thing "Run assessment" will use."""
        record = st.session_state.analysis
        if record:
            render_analysis(record)
            render_verdict(record, True)

    accepted = st.session_state.measured_bpm is not None

    if mode == "Upload recording":
        with st.container(border=True):
            st.markdown("**Recording**")
            st.caption(
                "Cover the rear camera lens with a fingertip, hold still, and "
                f"record 15–20 seconds with the flash on. The last {WINDOW_S:.0f} s "
                "are analysed, so the exposure settling at the start is skipped."
            )
            video = st.file_uploader(
                "Fingertip recording",
                type=["mp4", "mov", "avi", "webm"],
                label_visibility="collapsed",
            )

        if video:
            with st.spinner("Extracting plethysmograph…"):
                samples = extract_samples_from_video(video.getvalue())
            span = samples[-1].t - samples[0].t if len(samples) > 1 else 0.0

            if span < MIN_WINDOW_S:
                clear_reading()
                accepted = False
                render_idle("Recording too short")
                st.error(
                    f"Recording is too short to analyse — capture at least "
                    f"{MIN_WINDOW_S:.0f} seconds (this one is {span:.1f} s).",
                    icon=":material/error:",
                )
            else:
                estimate = estimate_pulse(samples)
                if estimate is None:
                    clear_reading()
                    accepted = False
                    render_idle("No pulse detected")
                    st.error(
                        "No clear pulse detected. Hold the fingertip still against "
                        "the lens for the whole recording.",
                        icon=":material/error:",
                    )
                else:
                    accepted = render_trace(
                        estimate, "Upload", estimate.bpm, estimate.quality
                    )
                    record = analysis_record(
                        estimate,
                        estimate.bpm,
                        estimate.quality,
                        "Upload",
                        "Accepted" if accepted else "Rejected",
                    )
                    if accepted:
                        store_reading(record)
                    else:
                        clear_reading()
                    render_analysis(record)
                    render_verdict(record, accepted)
        else:
            render_idle("Awaiting recording")
            render_captured()

    else:
        sampler = st.session_state.pulse_sampler
        tracker = st.session_state.pulse_tracker

        def camera_streamer():
            """Start the camera component, degrading to an error panel if it
            cannot initialise instead of taking the whole page down."""
            try:
                return webrtc_streamer(
                    key="pulse",
                    mode=WebRtcMode.SENDRECV,
                    # Every queued frame is sampled — the single-frame callback
                    # drops all but the newest when the worker falls behind.
                    queued_video_frames_callback=sampler.recv_queued,
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    },
                    media_stream_constraints={
                        "video": {
                            # Rear camera on phones (the one with the flash);
                            # laptops fall back to their only camera.
                            "facingMode": {"ideal": "environment"},
                            # 640x480 keeps the encoder at a steady 30 fps; the
                            # ROI mean needs pixels, not resolution.
                            "width": {"ideal": 640},
                            "height": {"ideal": 480},
                            "frameRate": {"ideal": 30},
                        },
                        "audio": False,
                    },
                    async_processing=True,
                )
            except Exception as exc:
                st.error(
                    f"Camera component could not start: {exc}",
                    icon=":material/videocam_off:",
                )
                return None

        with st.container(border=True):
            st.markdown("**Live camera**")
            st.markdown(
                "1. Press **Start** and allow camera access.  \n"
                "2. Press a fingertip flat over the lens so the picture turns "
                "uniformly red — nothing is measured until it does.  \n"
                f"3. Keep still. A first reading appears after {MIN_WINDOW_S:.0f} s "
                f"and is accepted once it holds steady over a full {WINDOW_S:.0f} s window."
            )
            ctx = camera_streamer()

        playing = bool(ctx is not None and ctx.state.playing)
        if playing and not st.session_state.live_was_playing:
            # Every new stream starts clean: no frames or estimates from a
            # previous attempt.
            sampler.clear()
            tracker.reset()
            st.session_state._pulse_was_locked = False
        st.session_state.live_was_playing = playing

        def reading_captured(state, estimate, status):
            """Accept only a locked reading computed on a full window that
            starts well after the finger was placed (exposure has settled)."""
            return (
                state.locked
                and estimate is not None
                and estimate.duration_s >= WINDOW_S - 0.5
                and status.seconds >= WINDOW_S + 1.5
            )

        def live_guidance(status, estimate, state, captured):
            """One line telling the user what to do next."""
            if captured:
                return (
                    "Reading captured — keep still for a steadier value, or run the assessment.",
                    "green",
                )
            if status.hint == "empty":
                return "Waiting for camera frames…", "gray"
            if status.hint == "uncovered":
                return (
                    "Lens not covered — press a fingertip flat over the camera lens.",
                    "yellow",
                )
            if status.hint == "dark":
                return (
                    "Too dark — turn towards a lamp or window so light passes through the fingertip.",
                    "yellow",
                )
            if status.hint == "saturated":
                return (
                    "Sensor saturated — ease the finger pressure or move away from the light.",
                    "yellow",
                )
            if estimate is None:
                return (
                    f"Acquiring — hold still, first reading after {MIN_WINDOW_S:.0f} s.",
                    "blue",
                )
            return {
                "acquiring": (
                    f"Hold still — accepted once the reading holds steady over {WINDOW_S:.0f} s.",
                    "blue",
                ),
                "stable": (
                    "Steady — keep the finger in place while the window fills.",
                    "blue",
                ),
                "holding": (
                    "Signal lost — holding the last value. Keep the finger still and fully over the lens.",
                    "yellow",
                ),
                "searching": (
                    "No reliable pulse — reposition the fingertip and hold still.",
                    "red",
                ),
            }[state.status]

        @st.fragment(run_every="0.5s")
        def live_monitor():
            """Refresh the monitor on its own clock without rerunning the page."""
            status = sampler.status()
            # A face or a room is not a plethysmograph: without a fingertip on
            # the lens nothing is estimated, so nothing spurious is shown.
            estimate = estimate_pulse(sampler.snapshot()) if status.covered else None
            state = tracker.update(
                time.monotonic(),
                bpm=estimate.bpm if estimate else None,
                quality=estimate.quality if estimate else 0.0,
            )
            captured = reading_captured(state, estimate, status)
            if captured:
                store_reading(
                    analysis_record(
                        estimate, state.bpm, estimate.quality, "Live", "Stable"
                    )
                )
                if not st.session_state._pulse_was_locked:
                    st.session_state._pulse_was_locked = True
                    st.rerun()  # the action row lives outside this fragment

            if estimate is None:
                if not status.covered:
                    render_idle("Lens not covered")
                else:
                    render_idle(
                        f"Acquiring · {min(status.seconds, MIN_WINDOW_S):.0f} of "
                        f"{MIN_WINDOW_S:.0f} s"
                    )
            else:
                badge = (
                    ("Captured", "green") if captured else STATUS_BADGES[state.status]
                )
                render_trace(
                    estimate, "Live", state.bpm, estimate.quality, status=badge
                )

            text, colour = live_guidance(status, estimate, state, captured)
            st.markdown(f":{colour}[{text}]")
            if status.covered and not captured:
                target = WINDOW_S + 1.5
                st.progress(
                    min(status.seconds / target, 1.0),
                    text=f"Steady signal · {min(status.seconds, target):.0f} / {target:.0f} s",
                )
            if sampler.errors:
                st.caption(f":red[{sampler.errors} camera frames could not be decoded]")
            render_captured()

        if playing:
            live_monitor()
        else:
            render_idle("Camera stopped")
            render_captured()

    with st.container(horizontal=True):
        if st.button("Back", icon=":material/arrow_back:"):
            reset_acquisition()
            goto(1)
        with st.container(horizontal=True, horizontal_alignment="right"):
            if mode == "Live camera" and st.session_state.analysis:
                if st.button(
                    "Restart reading",
                    icon=":material/restart_alt:",
                    help="Discard the captured reading and measure again.",
                ):
                    reset_acquisition()
                    st.rerun()
            if st.button(
                "Run assessment",
                type="primary",
                icon=":material/arrow_forward:",
                disabled=not accepted,
                shortcut="Mod+Enter",
                help="Keyboard shortcut: Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux). Requires an accepted pulse reading.",
            ):
                goto(3)

    if not accepted:
        if mode == "Live camera":
            st.caption(
                "Run assessment unlocks automatically once the live reading has "
                f"held steady over a full {WINDOW_S:.0f} s window."
            )
        else:
            st.caption("An accepted recording is required before assessment.")

# ==========================================================================
# STEP 3 — ASSESSMENT
# ==========================================================================
elif step == 3:
    st.subheader("Risk assessment")

    bmi_val = round(st.session_state.weight / (st.session_state.height**2), 2)
    pulse_rate = int(st.session_state.measured_bpm)
    bmi_label, bmi_colour = classify_bmi(bmi_val)
    hr_label, hr_colour = grade_hr(pulse_rate)

    with st.container(border=True):
        st.markdown("**Patient summary**")
        st.table(
            {
                "Age": f"{st.session_state.age} years",
                "Sex": st.session_state.gender,
                "BMI": f"{bmi_val} kg/m² · :{bmi_colour}-badge[{bmi_label}]",
                "Pulse": f"{pulse_rate} bpm · :{hr_colour}-badge[{hr_label}]",
                "Current smoker": st.session_state.current_smoker,
                "Signal quality": f"{(st.session_state.signal_pct or 0):.0f}%"
                f" · {st.session_state.source or '—'}",
            },
            border="horizontal",
            width="stretch",
        )

    if st.session_state.api_result is None:
        # Held in a placeholder so it can be cleared. Written directly, the
        # notice stayed on screen next to the finished results, because this
        # script run continues past the fetch rather than rerunning.
        cold_start_notice = st.empty()
        cold_start_notice.info(
            "**Scoring risk model** (30–45 seconds on first request)  \n"
            "The inference server may be starting up — this is normal.",
            icon=":material/hourglass_top:",
        )
        with st.spinner("Contacting risk model…"):
            params = {
                "sex": 1 if st.session_state.gender == "Male" else 0,
                "age": int(st.session_state.age),
                "pulse_rate": int(pulse_rate),
                "height": float(st.session_state.height),
                "weight": float(st.session_state.weight),
                "bmi": float(bmi_val),
                "pulse": float(pulse_rate),
                "ever_smoked": 1 if st.session_state.ever_smoked == "Yes" else 0,
                "current_smoker": 1 if st.session_state.current_smoker == "Yes" else 0,
            }
            try:
                # The model is on Cloud Run and scales to zero; a cold start can
                # take well over 15s, so allow for it rather than showing a
                # spurious "unreachable" error on the first request of the day.
                response = requests.get(API_URL, params=params, timeout=45)
                if response.status_code != 200:
                    st.error(
                        f"**Risk model error** (HTTP {response.status_code})  \n"
                        "The inference server may be temporarily unavailable. "
                        "Please try again in a moment.",
                        icon=":material/cloud_off:",
                    )
                    st.code(response.text)
                    if st.button("Retry assessment", icon=":material/refresh:"):
                        st.session_state.api_result = None
                        st.rerun()
                    st.stop()
                st.session_state.api_result = response.json()
            except requests.exceptions.Timeout:
                st.error(
                    "**Request timeout** (server took >45 seconds)  \n"
                    "The inference server is taking longer than expected. "
                    "Please try again — it may be faster on the second attempt.",
                    icon=":material/schedule:",
                )
                if st.button("Retry assessment", icon=":material/refresh:"):
                    st.session_state.api_result = None
                    st.rerun()
                st.stop()
            except Exception as e:
                st.error(
                    f"**Could not reach the risk model**: {e}  \n"
                    "Please check your internet connection and try again.",
                    icon=":material/cloud_off:",
                )
                if st.button("Retry assessment", icon=":material/refresh:"):
                    st.session_state.api_result = None
                    st.rerun()
                st.stop()
        # Results are in — retire the cold-start notice.
        cold_start_notice.empty()

    result = st.session_state.api_result

    diabetic_prob = round(float(result.get("diabetic", 0.0)) * 100, 1)
    hypertensive_prob = round(float(result.get("hypertensive", 0.0)) * 100, 1)
    diabetic_val = 1 if diabetic_prob >= 50 else 0
    hypertensive_val = 1 if hypertensive_prob >= 50 else 0

    # Risk comparison narrative — helps users understand the relationship
    elevated_count = diabetic_val + hypertensive_val
    if elevated_count == 0:
        risk_summary = "✓ **Your risk is within expected range** for both conditions"
        summary_color = "green"
    elif elevated_count == 1:
        condition = "diabetes" if diabetic_val else "hypertension"
        risk_summary = f"⚠️ **You have elevated risk for {condition}** — clinical follow-up advised"
        summary_color = "orange"
    else:
        risk_summary = "🔴 **Elevated risk for both conditions** — clinical follow-up strongly advised"
        summary_color = "red"

    def risk_panel(column, title, icon, probability):
        """Alarm-graded risk card: red above threshold, green below."""
        elevated = probability >= 50.0
        colour = "red" if elevated else "green"
        hexes = {"red": "#FB6A6A", "green": "#4ADE80"}
        with column:
            st.markdown(f"{icon} **{title}**")

            readout(
                "Estimated probability (%)", f"{probability:.1f}%", None, colour=colour
            )
            with st.skeleton(height=60):
                st.altair_chart(risk_bar(probability, hexes[colour]))
            if elevated:
                st.badge(
                    "Elevated — clinical follow-up advised",
                    icon=":material/priority_high:",
                    color="red",
                )
            else:
                st.badge(
                    "Within expected range", icon=":material/check:", color="green"
                )

    diabetes_col, hypertension_col = st.columns(2, border=True)
    risk_panel(diabetes_col, "Diabetes", ":material/water_drop:", diabetic_prob)
    risk_panel(
        hypertension_col, "Hypertension", ":material/monitor_heart:", hypertensive_prob
    )

    # Stated once beneath both panels rather than repeated inside each.
    st.caption("Bars span 0–100%; the dashed rule marks the 50% decision threshold.")

    # Risk summary narrative — explains the overall finding at a glance
    with st.container(border=True):
        st.markdown(risk_summary)

    st.subheader("Lifestyle guidance")

    if st.session_state.recommendation is None:
        # Caption and action on one row — a bare button under a lone line of
        # helper text left a conspicuous empty band here.
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(
                ":gray[Generates non-diagnostic lifestyle guidance from the "
                "screening result and patient record.]",
                width="content",
            )
            if st.button(
                "Generate guidance",
                type="secondary",
                icon=":material/auto_awesome:",
                help="Optional. Generate AI-powered lifestyle recommendations based on your screening results.",
            ):
                with st.spinner("Generating guidance…"):
                    st.session_state.recommendation = get_gemini_recommendation(
                        diabetic=diabetic_val,
                        hypertensive=hypertensive_val,
                        age=st.session_state.age,
                        bmi=bmi_val,
                        pulse=pulse_rate,
                        sex=st.session_state.gender,
                    )
                st.rerun()

    rec = st.session_state.recommendation

    if rec is not None:
        if "error" in rec:
            st.error(rec["error"], icon=":material/error:")
            if st.button("Retry", icon=":material/refresh:"):
                st.session_state.recommendation = None
                st.rerun()
        else:
            pop_context = rec.get("population_context", [])
            if pop_context:
                with st.container(border=True):
                    st.markdown(":material/groups: **Population Context**")
                    for tip in pop_context:
                        st.markdown(f"- {tip}")

            st.write("")

            action_sections = [
                ("Diet", ":material/nutrition:", rec.get("diet", [])),
                ("Exercise", ":material/directions_run:", rec.get("exercise", [])),
                ("Monitoring", ":material/vital_signs:", rec.get("monitoring", [])),
            ]

            cols = st.columns(len(action_sections), border=True, gap="medium")
            for column, (title, icon, tips) in zip(cols, action_sections):
                with column:
                    st.markdown(f"{icon} **{title}**")
                    for tip in tips:
                        st.markdown(f"- {tip}")

            links = rec.get("further_reading", [])
            if links:
                with st.container(border=True):
                    st.markdown(":material/menu_book: **Further reading**")
                    st.markdown("Trusted sources for additional information:")
                    for link in links:
                        st.markdown(f"- [{link['title']}]({link['url']})")

            if rec.get("disclaimer"):
                st.caption(rec["disclaimer"])

    with st.container(horizontal=True):
        if st.button("Back", icon=":material/arrow_back:"):
            goto(2)
        with st.container(horizontal=True, horizontal_alignment="right"):
            # Build export filename with patient ref if available
            patient_ref_slug = (
                st.session_state.patient_ref.replace(" ", "_")[:12] + "_"
                if st.session_state.patient_ref
                else ""
            )
            export_filename = (
                f"pulseiq_{patient_ref_slug}{st.session_state.session_id}.txt"
            )

            st.download_button(
                "Export summary",
                data=build_report(
                    bmi=bmi_val,
                    bsa=bsa_dubois(st.session_state.height, st.session_state.weight),
                    pulse=pulse_rate,
                    hr_label=hr_label,
                    quality=st.session_state.signal_pct or 0.0,
                    source=st.session_state.source or "—",
                    diabetic=diabetic_prob,
                    hypertensive=hypertensive_prob,
                    analysis=st.session_state.analysis,
                ),
                file_name=export_filename,
                mime="text/plain",
                icon=":material/download:",
                help="Download a plain-text clinical summary for medical records.",
            )
            if st.button("New session", type="primary", icon=":material/restart_alt:"):
                st.session_state._show_new_session_confirm = True

    # Confirmation dialog for new session
    if st.session_state.get("_show_new_session_confirm", False):

        @st.dialog("Start a new session?", width="small")
        def confirm_new_session():
            st.markdown(
                "This will clear all data from the current session. "
                "The exported report will not be affected."
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cancel", width="stretch"):
                    st.session_state._show_new_session_confirm = False
                    st.rerun()
            with col2:
                if st.button(
                    "Start new session",
                    type="primary",
                    width="stretch",
                    icon=":material/restart_alt:",
                ):
                    for key in ("measured_bpm", "signal_pct", "source"):
                        st.session_state[key] = None
                    st.session_state.patient_ref = ""
                    reset_acquisition()
                    st.session_state.session_id = uuid.uuid4().hex[:8].upper()
                    st.session_state._show_new_session_confirm = False
                    goto(1)

        confirm_new_session()
