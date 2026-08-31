import os
import time
import threading
import requests
import numpy as np
import streamlit as st
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

# ============ Live pulse measurement from camera ============
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
# ============ End of camera block ============

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
        response.raise_for_status()

        # Försök tolka svaret som JSON
        try:
            result = response.json()
        except requests.exceptions.JSONDecodeError:
            st.error(
                "API:et returnerade inte giltig JSON. Detta beror ofta på ett serverfel."
            )
            st.expander("Se rått API-svar (för felsökning)").code(response.text)
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
        st.error(f"Kunde inte nå API:et eller fick en felkod: {e}")
        # Om felet uppstod efter att vi fick ett svar (t.ex. 500-fel), visa svaret
        if "response" in locals() and response is not None:
            st.expander("Se serverns felsida").code(response.text)
