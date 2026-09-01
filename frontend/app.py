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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

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

st.title("🩺 PulseIQ – Diabetes Risk Prediction")

# ============ Session state ============
if "signal_buffer" not in st.session_state:
    st.session_state.signal_buffer = deque(maxlen=300)  # ~10s @ 30fps
if "measured_bpm" not in st.session_state:
    st.session_state.measured_bpm = None
lock = threading.Lock()
signal_buffer = st.session_state.signal_buffer


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


# ============ Tabs ============
tab1, tab2, tab3 = st.tabs(
    ["📝 1. Patient details", "❤️ 2. Pulse measurement", "📊 3. Analysis & Result"]
)

# ==========================================
# TAB 1: PATIENT DETAILS
# ==========================================
with tab1:
    st.markdown("### Enter patient information")

    col_demo, col_history = st.columns(2)

    with col_demo:
        st.markdown("**Demographics & Body**")
        age = st.number_input("Age", min_value=0, max_value=120, value=33)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.63)
        weight = st.number_input(
            "Weight (kg)", min_value=20.0, max_value=250.0, value=70.2
        )
        bmi = round(weight / (height**2), 2) if height > 0 else 0.0
        st.metric(label="Calculated BMI", value=bmi)

    with col_history:
        st.markdown("**Lifestyle**")
        ever_smoked = st.selectbox("Ever smoked?", ["No", "Yes"])
        current_smoker = st.selectbox("Current smoker?", ["No", "Yes"])
        st.caption("Pulse is measured from a video in tab 2.")

# ==========================================
# TAB 2: PULSE MEASUREMENT
# ==========================================
with tab2:
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

# ==========================================
# TAB 3: PREDICTION + GEMINI RECOMMENDATION
# ==========================================
with tab3:
    st.markdown("### Start risk assessment")

    pulse_rate = (
        int(st.session_state.measured_bpm)
        if st.session_state.get("measured_bpm")
        else None
    )

    if pulse_rate:
        st.info(
            f"**Ready for analysis:** BMI is {bmi} and measured pulse is {pulse_rate} bpm."
        )
    else:
        st.warning("Measure the pulse in tab 2 before running an analysis.")

    predict_clicked = st.button(
        "Predict Risk",
        type="primary",
        use_container_width=True,
        disabled=(pulse_rate is None),
    )

    if predict_clicked:
        # All 9 fields the API's PatientFeatures schema expects
        params = {
            "sex": 1 if gender == "Male" else 0,
            "age": int(age),
            "pulse_rate": int(pulse_rate),
            "height": float(height),
            "weight": float(weight),
            "bmi": float(bmi),
            "pulse": float(pulse_rate),
            "ever_smoked": 1 if ever_smoked == "Yes" else 0,
            "current_smoker": 1 if current_smoker == "Yes" else 0,
        }

        with st.spinner("Analysing via API..."):
            try:
                response = requests.get(API_URL, params=params)

                if response.status_code != 200:
                    st.error(f"API returned status {response.status_code}")
                    st.code(response.text)
                    st.stop()

                try:
                    result = response.json()
                except requests.exceptions.JSONDecodeError:
                    st.error("The API did not return valid JSON. Raw response below:")
                    st.code(response.text)
                    st.stop()

            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API: {e}")
                st.stop()

        st.success("Risk assessment complete!")

        # The API returns {"diabetic": 0/1, "hypertensive": 0/1, "cv": 0/1};
        # cardiovascular (cv) is intentionally not shown.
        diabetic = result.get("diabetic", 0)
        hypertensive_pred = result.get("hypertensive", 0)

        def label(v):
            return "⚠️ At risk" if str(v) in ("1", "yes", "Yes") else "✅ Not at risk"

        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Diabetes", label(diabetic))
        res_col2.metric("Hypertension", label(hypertensive_pred))

        with st.expander("Raw API response"):
            st.write(result)

        # -------- Gemini recommendation --------
        st.markdown("---")
        st.markdown("### 🤖 Personalised recommendation")

        with st.spinner("Generating recommendation..."):
            rec = get_gemini_recommendation(
                diabetic=diabetic,
                hypertensive=hypertensive_pred,
                age=age,
                bmi=bmi,
                pulse=pulse_rate,
            )

        if "error" in rec:
            st.error(rec["error"])
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
