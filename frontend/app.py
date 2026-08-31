import os
import time
import threading
import requests
import numpy as np
import streamlit as st
import imageio.v3 as iio
from collections import deque
from scipy.signal import butter, filtfilt
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(
    page_title="PulseIQ – Diabetes Risk", page_icon="🩺", layout="centered"
)

API_URL = os.environ.get(
    "API_URL", "https://pulseiq-api-431111687933.europe-west1.run.app/predict"
)

st.title("🩺 PulseIQ – Diabetes Risk Prediction")
st.markdown("Enter the patient's details to get a prediction from the model.")

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
    """Estimates heart rate (bpm) from the brightness signal via bandpass + FFT."""
    signal = np.array(signal, dtype=np.float32)
    if len(signal) < fps * 3:
        return None
    signal = signal - signal.mean()
    low, high, nyq = 0.7, 4.0, fps / 2.0
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    filtered = filtfilt(b, a, signal)
    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
    valid = (freqs >= low) & (freqs <= high)
    if not valid.any():
        return None
    return round(freqs[valid][np.argmax(fft[valid])] * 60.0, 1)


@st.cache_data(show_spinner=False)
def extract_signal_from_video(video_bytes, max_frames=300):
    """Reads up to max_frames of red-channel brightness from the center of each frame.

    Returns (signal, fps). imageio gives RGB, so the red channel is index 0.
    Only a small center crop is averaged (where the fingertip covers the lens),
    which is faster and gives a cleaner pulse signal.
    """
    # Detect fps from metadata, fall back to 30
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


# ============ Measure pulse from an uploaded video ============
st.subheader("📤 Measure pulse from an uploaded video")
st.caption(
    "Upload a video where a fingertip covers the camera lens. "
    "The red-channel signal is extracted, plotted, and turned into a pulse."
)
video = st.file_uploader("Upload a fingertip video", type=["mp4", "mov", "avi", "webm"])

if video:
    with st.spinner("Analysing video..."):
        # Pass raw bytes so @st.cache_data can hash the input
        signal, fps = extract_signal_from_video(video.getvalue())

    if len(signal) < 10:
        st.warning("The video seems too short to analyse.")
    else:
        st.line_chart(signal)  # the PPG graph
        st.caption(f"Read {len(signal)} frames at ~{fps:.0f} fps")
        bpm = compute_bpm(signal, fps=fps)
        if bpm:
            st.session_state.measured_bpm = bpm
            st.metric("Measured pulse", f"{bpm} bpm")
        else:
            st.warning(
                "Couldn't detect a clear pulse — hold the fingertip still on the "
                "lens for a few seconds, or the lighting may have shifted too much."
            )
else:
    # ============ Live camera – only shown when NO video is uploaded ============
    with st.expander("📹 Measure pulse from camera (optional)", expanded=False):
        st.caption(
            "Cover the camera lens with your fingertip and hold still. "
            "The measured pulse will fill in the Pulse rate field below."
        )
        col_cam, col_graph = st.columns(2)
        with col_cam:
            ctx = webrtc_streamer(
                key="pulse",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=video_frame_callback,
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                    ]
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
                    chart_ph.line_chart(data)
                    bpm = compute_bpm(data)
                    if bpm:
                        st.session_state.measured_bpm = bpm
                        bpm_ph.metric("Measured pulse", f"{bpm} bpm")
                time.sleep(0.3)
# ============ End of pulse measurement ============

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=42)
    gender = st.selectbox("Gender", ["Female", "Male"])
    # Default value becomes the measured pulse if available
    default_pulse = (
        int(st.session_state.measured_bpm) if st.session_state.measured_bpm else 66
    )
    pulse_rate = st.number_input(
        "Pulse rate", min_value=30, max_value=200, value=default_pulse
    )
    systolic_bp = st.number_input("Systolic BP", min_value=70, max_value=250, value=110)
    diastolic_bp = st.number_input(
        "Diastolic BP", min_value=40, max_value=150, value=73
    )
    glucose = st.number_input("Glucose", min_value=2.0, max_value=30.0, value=5.88)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.65)
with col2:
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.2)
    family_diabetes = st.selectbox("Family history of diabetes", ["No", "Yes"])
    hypertensive = st.selectbox("Hypertensive", ["No", "Yes"])
    family_hypertension = st.selectbox("Family history of hypertension", ["No", "Yes"])
    cardiovascular_disease = st.selectbox("Cardiovascular disease", ["No", "Yes"])
    stroke = st.selectbox("Stroke", ["No", "Yes"])

# BMI is calculated automatically from height and weight
bmi = round(weight / (height**2), 2) if height > 0 else 0.0
st.metric("BMI (auto-calculated)", bmi)

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
    "cardiovascular_disease": 1 if cardiovascular_disease == "Yes" else 0,
    "stroke": 1 if stroke == "Yes" else 0,
}

if st.button("Predict risk", type="primary"):
    try:
        response = requests.get(API_URL, params=params)

        # Show what the server actually returned (helps diagnose non-JSON errors)
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

        pred = result.get("diabetic_prediction")
        if pred is not None:
            if str(pred).lower() in ("yes", "1"):
                st.error("Prediction: Diabetic")
            else:
                st.success("Prediction: Not diabetic")

        risk = result.get("diabetic_risk")
        if risk is not None:
            st.metric("Estimated risk", f"{risk:.1%}")

        with st.expander("Raw API response"):
            st.write(result)

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API: {e}")
