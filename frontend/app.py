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


@st.cache_data(show_spinner=False)
def get_gemini_recommendation(diabetic, hypertensive, age, bmi, glucose, pulse):
    """Ask Gemini (fast Flash-Lite, thinking off) for a lifestyle recommendation.

    Cached: identical inputs return instantly without another API call. Scalar
    args (not a dict) so Streamlit can hash them for the cache. The API key is
    sent in a header (never in the URL). Retries on transient 503/429.
    """
    if not GEMINI_API_KEY:
        return "⚠️ No GEMINI_API_KEY found. Add it to your .env (local) or Streamlit Secrets (cloud)."

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash-lite:generateContent"
    )
    headers = {"x-goog-api-key": GEMINI_API_KEY}

    prompt = f"""You are a helpful health assistant. Based on the data below, write a
short, friendly set of lifestyle recommendations (diet, exercise, monitoring, and
when to see a doctor). Do NOT give a diagnosis. Keep it to 4-6 concise bullet points
and add a one-line disclaimer that this is not medical advice.

Patient: age {age}, BMI {bmi}, glucose {glucose}, resting pulse {pulse} bpm.
Model predictions (1 = at risk, 0 = not at risk):
- Diabetic: {diabetic}
- Hypertensive: {hypertensive}
"""

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 400,
            "thinkingConfig": {"thinkingBudget": 0},  # 0 = no thinking -> fastest
        },
    }

    for attempt in range(3):  # retry on transient overload / rate limit
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            if status in (503, 429) and attempt < 2:
                time.sleep(2)
                continue
            return "The recommendation service is busy right now. Please try again in a moment."
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
        ever_smoked = st.selectbox("Ever smoked?", ["Yes", "No"])
        current_smoker = st.selectbox("Current smoker?", ["Yes", "No"])

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
    # 1. Capture the inputs from your UI first (Streamlit example)

    if predict_clicked:
        try:
            params = {
                "sex": 1 if gender == "Male" else 0,
                "age": int(age or 0),
                "pulse_rate": int(pulse_rate or 0),
                "height": float(height or 0.0),
                "weight": float(weight or 0.0),
                "bmi": float(bmi or 0.0),
                "pulse": float(pulse or 0.0),
                "ever_smoked": 1 if ever_smoked == "Yes" else 0,
                "current_smoker": 1 if current_smoker == "Yes" else 0,
            }
            # Run your prediction here
        except ValueError:
            print("Please ensure all numeric fields are filled out correctly.")

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

        # -------- Gemini recommendation (fast, cached) --------
        st.markdown("---")
        st.markdown("### 🤖 Personalised recommendation")

        with st.spinner("Generating recommendation..."):
            recommendation = get_gemini_recommendation(
                diabetic=diabetic,
                hypertensive=hypertensive_pred,
                age=age,
                bmi=bmi,
                glucose=glucose,
                pulse=pulse_rate,
            )

        st.markdown(recommendation)
