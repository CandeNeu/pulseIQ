import os
import time
import threading
import requests
import numpy as np
import pandas as pd
import streamlit as st
import imageio.v3 as iio
from collections import deque
from scipy.signal import butter, filtfilt
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import json

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


API_URL = os.environ.get(
    "API_URL", "https://pulseiq-api-431111687933.europe-west1.run.app/predict"
)

# ============ Page config and CSS ============
st.set_page_config(page_title="PulseIQ", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
        margin: 0 auto;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #FF4B4B;
    }
    .stButton>button {
        border-radius: 8px;
        height: 3rem;
        font-size: 1.1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ============ Session state ============
if "step" not in st.session_state:
    st.session_state.step = 1

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

if "age" not in st.session_state:
    st.session_state.age = 25
if "gender" not in st.session_state:
    st.session_state.gender = "Male"
if "height" not in st.session_state:
    st.session_state.height = 1.70
if "weight" not in st.session_state:
    st.session_state.weight = 68.90
if "ever_smoked" not in st.session_state:
    st.session_state.ever_smoked = "No"
if "current_smoker" not in st.session_state:
    st.session_state.current_smoker = "No"

if "signal_buffer" not in st.session_state:
    st.session_state.signal_buffer = deque(maxlen=300)  # ~10s @ 30fps
if "measured_bpm" not in st.session_state:
    st.session_state.measured_bpm = None
if "api_result" not in st.session_state:
    st.session_state.api_result = None
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None
lock = threading.Lock()
signal_buffer = st.session_state.signal_buffer

# ============ Session CSS ============
is_dark = st.session_state.theme_mode == "Dark"
bg_color = "#0e1117" if is_dark else "#f8f9fa"
card_bg = "#1e232d" if is_dark else "#ffffff"
text_color = "#ffffff" if is_dark else "#111827"
subtext_color = "#94a3b8" if is_dark else "#6b7280"
border_color = "#e5e7eb" if is_dark else "#e5e7eb"

st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}

    .block-container {{
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        margin: 0 auto;
    }}

    .risk-card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}

    .badge-safe {{
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid #22c55e;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }}

    .badge-risk {{
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }}

    .stButton>button {{
        border-radius: 8px;
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def video_frame_callback(frame):
    """Runs on every camera frame – extracts red brightness (PPG signal)."""
    img = frame.to_ndarray(format="bgr24")
    with lock:
        signal_buffer.append(float(img[:, :, 2].mean()))
    return frame


def compute_bpm(signal, fps=30.0):
    """Estimate heart rate from a brightness signal via bandpass + FFT.

    Returns (bpm, filtered_signal, freqs_bpm, spectrum).
    """
    signal = np.array(signal, dtype=np.float32)
    if len(signal) < fps * 3:
        return None, None, None, None
    signal = signal - signal.mean()
    low, high, nyq = 0.7, 4.0, fps / 2.0
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    filtered = filtfilt(b, a, signal)

    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
    valid = (freqs >= low) & (freqs <= high)
    if not valid.any():
        return None, filtered, None, None

    bpm = round(freqs[valid][np.argmax(fft[valid])] * 60.0, 1)
    freqs_bpm = freqs[valid] * 60.0
    spectrum = fft[valid]
    return bpm, filtered, freqs_bpm, spectrum


@st.cache_data(show_spinner=False)
def extract_signal_from_video(video_bytes, max_frames=300):
    """Read up to max_frames of red-channel brightness from the center of each frame.

    Returns (signal, fps). imageio gives RGB, so the red channel is index 0.
    """
    try:
        meta = iio.immeta(video_bytes, plugin="pyav")
        fps = float(meta.get("fps", 30.0))
    except Exception:
        fps = 30.0

    signal = []
    for frame in iio.imiter(video_bytes, plugin="pyav"):
        h, w, _ = frame.shape
        cy, cx = h // 2, w // 2
        crop = frame[cy - 50 : cy + 50, cx - 50 : cx + 50, 0]  # center, red channel
        signal.append(float(crop.mean()))
        if len(signal) >= max_frames:
            break
    return signal, fps


def get_gemini_recommendation(diabetic, hypertensive, age, bmi, pulse):
    """Ask Gemini for a structured lifestyle recommendation.

    Returns a dict with keys: diet, exercise, monitoring (each a list of strings),
    further_reading (list of {title, url}) and disclaimer. On error returns a dict
    with an 'error' key instead.
    """
    if not GEMINI_API_KEY:
        return {
            "error": "⚠️ No GEMINI_API_KEY found. Add it to your .env (local) or Streamlit Secrets (cloud)."
        }

    safe_api_key = GEMINI_API_KEY.strip()

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash:generateContent"
    )
    headers = {"x-goog-api-key": safe_api_key, "Content-Type": "application/json"}

    diabetes_status = "Elevated risk" if diabetic == 1 else "Standard risk"
    ht_status = "Elevated risk" if hypertensive == 1 else "Standard risk"

    system_instruction = """You are an empathetic, knowledgeable health and lifestyle educator.
Return lifestyle recommendations as JSON ONLY — no markdown, no code fences, no text outside the JSON object.

The JSON must have exactly this shape:
{
  "diet": ["detailed tip 1", "detailed tip 2"],
  "exercise": ["detailed tip 1", "detailed tip 2"],
  "monitoring": ["detailed tip 1", "detailed tip 2"],
  "further_reading": [{"title": "WHO – Healthy diet", "url": "https://www.who.int/..."}],
  "disclaimer": "one-line disclaimer text"
}

RULES:
1. Each list should have 2-4 detailed, verbose tips that explain the reasoning, not just the instruction.
2. Where relevant, include concrete statistics (e.g. 150 minutes/week of moderate activity, healthy BMI range 18.5-24.9) and briefly attribute the source type (WHO, CDC, ADA, NHS).
3. "further_reading" must contain 3-5 links using only these real domains: who.int, cdc.gov, diabetes.org, nhs.uk, mayoclinic.org. Prefer main topic pages; do not invent deep URLs.
4. NEVER provide a medical diagnosis or diagnostic language.
5. The "disclaimer" value must be exactly: "This is an automated lifestyle suggestion based on statistical data, not medical advice. Always consult a healthcare professional."
6. Output valid JSON only.
"""

    user_prompt = f"""Patient Profile:
- Age: {age}
- BMI: {bmi}
- Resting Pulse: {pulse} bpm

Machine Learning Risk Assessment:
- Diabetes: {diabetes_status}
- Hypertension: {ht_status}

Generate the recommendations as JSON."""

    body = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 1500,
            "temperature": 0.3,
            "responseMimeType": "application/json",  # force strict JSON
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
            return {"error": f"**API Error {status}:** `{error_details}`"}
        except (KeyError, IndexError):
            return {"error": "Gemini returned an unexpected response format."}
        except json.JSONDecodeError:
            return {"error": "Gemini did not return valid JSON. Try again."}


# ============ Header + step indicator ============
header_col, theme_col = st.columns([5, 1])

with header_col:
    st.title("🩺 PulseIQ – Risk Assessment")

with theme_col:
    selected_theme = st.selectbox(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme_mode == "Dark" else 1,
        label_visibility="collapsed",
    )
    if selected_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = selected_theme
        st.rerun()

step_cols = st.columns(3)
step_labels = ["1. Patient Details", "2. Pulse Measurement", "3. Analysis & Results"]
for i, label in enumerate(step_labels, 1):
    with step_cols[i - 1]:
        if st.session_state.step == i:
            st.markdown(f"**{label}**")
        elif st.session_state.step > i:
            st.markdown(f"{label}")
        else:
            st.markdown(
                f"<span style='color:{subtext_color}'>{label}</span>",
                unsafe_allow_html=True,
            )

st.divider()


# ==========================================
# STEP 1: PATIENT DETAILS
# ==========================================
if st.session_state.step == 1:
    st.subheader("📝 Step 1: Patient Information")

    col_demo, col_vitals, col_history = st.columns(3)

    with col_demo:
        age = st.number_input(
            "Age", min_value=0, max_value=120, value=st.session_state.age
        )
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0 if st.session_state.gender == "Male" else 1,
        )

    with col_vitals:
        height = st.number_input(
            "Height (m)",
            min_value=1.0,
            max_value=2.5,
            value=st.session_state.height,
            step=0.01,
        )
        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=250.0,
            value=st.session_state.weight,
            step=0.5,
        )
        bmi = round(weight / (height**2), 2) if height > 0 else 0.0
        st.metric(label="Calculated BMI", value=bmi)

    with col_history:
        ever_smoked = st.selectbox(
            "Ever smoked?",
            ["No", "Yes"],
            index=0 if st.session_state.ever_smoked == "No" else 1,
        )
        current_smoker = st.selectbox(
            "Current smoker?",
            ["No", "Yes"],
            index=0 if st.session_state.current_smoker == "No" else 1,
        )

    _, next_col = st.columns([4, 1])
    with next_col:
        if st.button("Next: Pulse", type="primary", use_container_width=True):
            st.session_state.age = age
            st.session_state.gender = gender
            st.session_state.height = height
            st.session_state.weight = weight
            st.session_state.ever_smoked = ever_smoked
            st.session_state.current_smoker = current_smoker
            st.session_state.step = 2
            st.rerun()

# ==========================================
# STEP 2: PULSE MEASUREMENT
# ==========================================
elif st.session_state.step == 2:
    st.markdown("### Upload a video for pulse measurement")

    col_upload, col_metric = st.columns([2, 1])

    with col_upload:
        video = st.file_uploader(
            "Upload a fingertip video",
            type=["mp4", "mov", "avi", "webm"],
            label_visibility="collapsed",
        )

    if video:
        with st.spinner("Analysing video..."):
            signal, fps = extract_signal_from_video(video.getvalue())

        if len(signal) < 10:
            st.warning("The video seems too short to analyse.")
        else:
            bpm, filtered, freqs_bpm, spectrum = compute_bpm(signal, fps=fps)

            if bpm:
                st.session_state.measured_bpm = bpm
                with col_metric:
                    st.metric(label="Measured pulse", value=f"{bpm} bpm")

                graph_col1, graph_col2 = st.columns(2)
                with graph_col1:
                    st.markdown("#### ❤️ Filtered pulse signal")
                    st.caption(
                        "Fingertip brightness over time, with slow drift removed. "
                        "Each peak is one heartbeat."
                    )
                    st.line_chart(filtered)
                with graph_col2:
                    st.markdown("#### 📊 Frequency spectrum")
                    st.caption(
                        f"How strong each possible heart rate is in the signal. "
                        f"The tall peak marks the pulse ({bpm:.0f} bpm)."
                    )
                    spectrum_df = pd.DataFrame({"bpm": freqs_bpm, "strength": spectrum})
                    st.line_chart(spectrum_df, x="bpm", y="strength")
            else:
                st.warning(
                    "Couldn't detect a clear pulse — hold the fingertip still "
                    "on the lens for a few seconds."
                )

    with st.expander("📹 Or measure from the live camera instead"):
        st.caption("Cover the lens with your fingertip and hold still.")
        col_cam, col_graph = st.columns(2)
        with col_cam:
            ctx = webrtc_streamer(
                key="pulse",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=video_frame_callback,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        with col_graph:
            chart_ph = st.empty()
            bpm_ph = st.empty()

        if ctx.state.playing:
            while ctx.state.playing:
                with lock:
                    data = list(signal_buffer)
                if len(data) > 5:
                    bpm, filtered, _, _ = compute_bpm(data)
                    if filtered is not None:
                        chart_ph.line_chart(filtered)
                    if bpm:
                        st.session_state.measured_bpm = bpm
                        bpm_ph.metric("Measured pulse", f"{bpm} bpm")
                time.sleep(0.3)

    back_col, _, next_col = st.columns([1, 4, 1])
    with back_col:
        if st.button("⬅️ Back"):
            st.session_state.step = 1
            st.rerun()

    with next_col:
        has_pulse = st.session_state.measured_bpm is not None
        if st.button(
            "Next: Analysis ➔",
            type="primary",
            disabled=not has_pulse,
            use_container_width=True,
        ):
            st.session_state.step = 3
            st.session_state.api_result = None
            st.session_state.recommendation = None
            st.rerun()

# ==========================================
# STEP 3: PREDICTION + GEMINI RECOMMENDATION
# ==========================================
elif st.session_state.step == 3:
    st.subheader("📊 Step 3: Analysis & Prediction")

    bmi_val = round(st.session_state.weight / (st.session_state.height**2), 2)
    pulse_rate = int(st.session_state.measured_bpm)

    st.markdown(
        f"""
        <div class="risk-card">
            <strong>Patient Summary:</strong>
            Age: <code>{st.session_state.age}</code> |
            Sex: <code>{st.session_state.gender}</code> |
            BMI: <code>{bmi_val} kg/m²</code> |
            Pulse: <code>{pulse_rate} bpm</code> |
            Smoker: <code>{st.session_state.current_smoker}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.api_result is None:
        with st.spinner("Analysing risk via API..."):
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
                response = requests.get(API_URL, params=params, timeout=15)
                if response.status_code != 200:
                    st.error(f"API returned status {response.status_code}")
                    st.code(response.text)
                    st.stop()
                st.session_state.api_result = response.json()
            except Exception as e:
                st.error(f"Could not reach the API: {e}")
                st.stop()

    result = st.session_state.api_result
    st.success("Risk assessment complete!")

    # API returns the probability of class 0 as a decimal (e.g. 0.99).
    # Risk = probability of the "at risk" class = 1 - that value, as a percent.
    diabetic_raw = float(result.get("diabetic", 0.0))
    hypertensive_raw = float(result.get("hypertensive", 0.0))

    diabetic_prob = round((diabetic_raw) * 100, 1)
    hypertensive_prob = round((hypertensive_raw) * 100, 1)

    diabetic_val = 1 if diabetic_prob >= 50 else 0
    hypertensive_val = 1 if hypertensive_prob >= 50 else 0

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        is_diab_risk = diabetic_prob >= 50.0
        badge_html = (
            "<span class='badge-risk'>⚠️ At Risk</span>"
            if is_diab_risk
            else "<span class='badge-safe'>✅ Low Risk</span>"
        )
        val_color = "#ef4444" if is_diab_risk else "#22c55e"

        st.markdown(
            f"""
            <div class="risk-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">🩸 Diabetes</h3>
                    {badge_html}
                </div>
                <h1 style="color:{val_color}; margin: 12px 0 4px 0; font-size:2.8rem;">{diabetic_prob:.1f}%</h1>
                <p style="color:{subtext_color}; font-size:0.85rem; margin:0;">Estimated risk probability</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(min(max(diabetic_prob / 100.0, 0.0), 1.0)))

    with res_col2:
        is_hyp_risk = hypertensive_prob >= 50.0
        badge_html = (
            "<span class='badge-risk'>⚠️ At Risk</span>"
            if is_hyp_risk
            else "<span class='badge-safe'>✅ Low Risk</span>"
        )
        val_color = "#ef4444" if is_hyp_risk else "#22c55e"

        st.markdown(
            f"""
            <div class="risk-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">🫀 Hypertension</h3>
                    {badge_html}
                </div>
                <h1 style="color:{val_color}; margin: 12px 0 4px 0; font-size:2.8rem;">{hypertensive_prob:.1f}%</h1>
                <p style="color:{subtext_color}; font-size:0.85rem; margin:0;">Estimated risk probability</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(min(max(hypertensive_prob / 100.0, 0.0), 1.0)))

    # with st.expander("Raw API response"):
    #    st.write(result)

    # -------- Gemini recommendation --------
    st.markdown("---")
    st.markdown("### 🤖 Personalised recommendation")

    if st.session_state.recommendation is None:
        if st.button("✨ Get recommendation", type="primary", use_container_width=True):
            with st.spinner("Generating recommendation..."):
                st.session_state.recommendation = get_gemini_recommendation(
                    diabetic=diabetic_val,
                    hypertensive=hypertensive_val,
                    age=st.session_state.age,
                    bmi=bmi_val,
                    pulse=pulse_rate,
                )
            st.rerun()

    rec = st.session_state.recommendation

    if rec is not None:
        if "error" in rec:
            st.error(rec["error"])
            if st.button("🔄 Retry", type="primary", use_container_width=True):
                with st.spinner("Retrying recommendation..."):
                    st.session_state.recommendation = get_gemini_recommendation(
                        diabetic=diabetic_val,
                        hypertensive=hypertensive_val,
                        age=st.session_state.age,
                        bmi=bmi_val,
                        pulse=pulse_rate,
                    )
                st.rerun()
        else:
            col_diet, col_exercise, col_monitor = st.columns(3)

            with col_diet:
                st.markdown("#### 🥗 Diet")
                for tip in rec.get("diet", []):
                    st.markdown(f"- {tip}")

            with col_exercise:
                st.markdown("#### 🏃 Exercise")
                for tip in rec.get("exercise", []):
                    st.markdown(f"- {tip}")

            with col_monitor:
                st.markdown("#### 📈 Monitoring")
                for tip in rec.get("monitoring", []):
                    st.markdown(f"- {tip}")

            links = rec.get("further_reading", [])
            if links:
                st.markdown("---")
                st.markdown("#### 📚 Further reading")
                st.markdown(" · ".join(f"[{l['title']}]({l['url']})" for l in links))

            if rec.get("disclaimer"):
                st.caption(f"_{rec['disclaimer']}_")

    # Navigation
    st.markdown("<br>", unsafe_allow_html=True)
    back_col, _, reset_col = st.columns([1, 3, 1])
    with back_col:
        if st.button("⬅️ Back"):
            st.session_state.step = 2
            st.rerun()

    with reset_col:
        if st.button("🔄 Start New Test", type="primary", use_container_width=True):
            st.session_state.step = 1
            st.session_state.measured_bpm = None
            st.session_state.api_result = None
            st.session_state.recommendation = None
            st.rerun()
