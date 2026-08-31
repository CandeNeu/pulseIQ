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
from dotenv import load_dotenv, find_dotenv

# ============ Environment / secrets ============
load_dotenv(find_dotenv())  # finds .env even from the frontend/ folder; never commit it
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

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

    /* Constrain the content width and add breathing room on the sides */
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


def get_gemini_recommendation(patient, predictions):
    """Ask Gemini for a lifestyle recommendation based on the patient + predictions.

    Returns the recommendation text, or an error string. The API key is sent in a
    header (never in the URL) so it can't leak into error messages.
    """
    if not GEMINI_API_KEY:
        return "⚠️ No GEMINI_API_KEY found. Add it to your .env (local) or Streamlit Secrets (cloud)."

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.6-flash:generateContent"
    )
    headers = {"x-goog-api-key": GEMINI_API_KEY}

    prompt = f"""You are a helpful health assistant. Based on the patient data and
model predictions below, write a short, friendly set of lifestyle recommendations
(diet, exercise, monitoring, and when to see a doctor). Do NOT give a diagnosis.
Keep it to 4-6 concise bullet points and add a one-line disclaimer that this is not
medical advice.

Patient data:
{patient}

Model predictions (1 = at risk, 0 = not at risk):
- Diabetic: {predictions['diabetic']}
- Hypertensive: {predictions['hypertensive']}
- Cardiovascular: {predictions['cv']}
"""

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 500},
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.RequestException as e:
        # TEMPORARY DEBUG: show the real reason (status + body), never the key
        status = getattr(e.response, "status_code", "no response")
        body_text = getattr(e.response, "text", str(e))
        return f"Could not reach Gemini (status {status}): {body_text}"
    except (KeyError, IndexError):
        return "Gemini returned an unexpected response."


# ============ Tabs ============
tab1, tab2, tab3 = st.tabs(
    ["📝 1. Patient details", "❤️ 2. Pulse measurement", "📊 3. Analysis & Result"]
)

# ==========================================
# TAB 1: PATIENT DETAILS
# ==========================================
with tab1:
    st.markdown("### Enter patient information")

    col_demo, col_vitals, col_history = st.columns(3)

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

    with col_vitals:
        st.markdown("**Vital signs**")
        systolic_bp = st.number_input("Systolic BP", value=106)
        diastolic_bp = st.number_input("Diastolic BP", value=67)
        glucose = st.number_input("Glucose", value=5.81)

    with col_history:
        st.markdown("**Medical history**")
        family_diabetes = st.selectbox("Family history of diabetes", ["No", "Yes"])
        hypertensive = st.selectbox("Hypertensive", ["No", "Yes"])
        family_hypertension = st.selectbox(
            "Family history of hypertension", ["No", "Yes"]
        )
        cardiovascular = st.selectbox("Cardiovascular disease", ["No", "Yes"])
        stroke = st.selectbox("Stroke", ["No", "Yes"])

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
        params = {
            "age": age,
            "gender": gender,
            "pulse_rate": pulse_rate,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "glucose": glucose,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "family_diabetes": 1 if family_diabetes == "Yes" else 0,
            "hypertensive": 1 if hypertensive == "Yes" else 0,
            "family_hypertension": 1 if family_hypertension == "Yes" else 0,
            "cardiovascular_disease": 1 if cardiovascular == "Yes" else 0,
            "stroke": 1 if stroke == "Yes" else 0,
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

        # The API returns: {"diabetic": 0/1, "hypertensive": 0/1, "cv": 0/1}
        predictions = {
            "diabetic": result.get("diabetic", 0),
            "hypertensive": result.get("hypertensive", 0),
            "cv": result.get("cv", 0),
        }

        def label(v):
            return "⚠️ At risk" if str(v) in ("1", "yes", "Yes") else "✅ Not at risk"

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Diabetes", label(predictions["diabetic"]))
        res_col2.metric("Hypertension", label(predictions["hypertensive"]))
        res_col3.metric("Cardiovascular", label(predictions["cv"]))

        with st.expander("Raw API response"):
            st.write(result)

        # -------- Gemini recommendation --------
        st.markdown("---")
        st.markdown("### 🤖 Personalised recommendation")

        with st.spinner("Generating recommendation..."):
            recommendation = get_gemini_recommendation(params, predictions)

        st.markdown(recommendation)
