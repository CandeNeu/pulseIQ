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

st.set_page_config(
    page_title="PulseIQ – Diabetes Risk", page_icon="🩺", layout="centered"
)

API_URL = os.environ.get(
    "API_URL", "https://pulseiq-api-431111687933.europe-west1.run.app/predict"
)

st.title("🩺 PulseIQ – Diabetes Risk Prediction")
st.markdown(
    "**Step 1** – fill in every patient detail. "
    "**Step 2** – measure the pulse from a video. "
    "**Step 3** – get the prediction."
)

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


# =====================================================================
# STEP 1 — Patient details (all fields start empty and are required)
# =====================================================================
st.header("1️⃣ Patient details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        "Age", min_value=0, max_value=120, value=None, placeholder="Enter age"
    )
    gender = st.selectbox(
        "Gender", ["Female", "Male"], index=None, placeholder="Select gender"
    )
    systolic_bp = st.number_input(
        "Systolic BP",
        min_value=70,
        max_value=250,
        value=None,
        placeholder="Enter value",
    )
    diastolic_bp = st.number_input(
        "Diastolic BP",
        min_value=40,
        max_value=150,
        value=None,
        placeholder="Enter value",
    )
    glucose = st.number_input(
        "Glucose", min_value=2.0, max_value=30.0, value=None, placeholder="Enter value"
    )
    height = st.number_input(
        "Height (m)",
        min_value=1.0,
        max_value=2.5,
        value=None,
        placeholder="Enter height",
    )
with col2:
    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=250.0,
        value=None,
        placeholder="Enter weight",
    )
    family_diabetes = st.selectbox(
        "Family history of diabetes", ["No", "Yes"], index=None, placeholder="Select"
    )
    hypertensive = st.selectbox(
        "Hypertensive", ["No", "Yes"], index=None, placeholder="Select"
    )
    family_hypertension = st.selectbox(
        "Family history of hypertension",
        ["No", "Yes"],
        index=None,
        placeholder="Select",
    )
    cardiovascular_disease = st.selectbox(
        "Cardiovascular disease", ["No", "Yes"], index=None, placeholder="Select"
    )
    stroke = st.selectbox("Stroke", ["No", "Yes"], index=None, placeholder="Select")

# BMI is calculated automatically once height and weight are entered
if height and weight:
    bmi = round(weight / (height**2), 2)
    st.metric("BMI (auto-calculated)", bmi)
else:
    bmi = None
    st.caption("BMI will be calculated once height and weight are entered.")

# All fields must be filled before the video unlocks
required_fields = {
    "Age": age,
    "Gender": gender,
    "Systolic BP": systolic_bp,
    "Diastolic BP": diastolic_bp,
    "Glucose": glucose,
    "Height": height,
    "Weight": weight,
    "Family history of diabetes": family_diabetes,
    "Hypertensive": hypertensive,
    "Family history of hypertension": family_hypertension,
    "Cardiovascular disease": cardiovascular_disease,
    "Stroke": stroke,
}
missing = [name for name, val in required_fields.items() if val is None]
details_done = len(missing) == 0

# =====================================================================
# STEP 2 — Measure pulse (locked until every field above is filled)
# =====================================================================
st.header("2️⃣ Measure pulse")

if not details_done:
    st.warning(
        "Please fill in all patient details to unlock pulse measurement. "
        "Still missing: " + ", ".join(missing)
    )
else:
    st.caption(
        "Upload a video where a fingertip covers the camera lens. "
        "The red-channel signal is extracted, filtered, and the heart frequency plotted."
    )
    video = st.file_uploader(
        "Upload a fingertip video", type=["mp4", "mov", "avi", "webm"]
    )

    if video:
        with st.spinner("Analysing video..."):
            signal, fps = extract_signal_from_video(video.getvalue())

        if len(signal) < 10:
            st.warning("The video seems too short to analyse.")
        else:
            bpm, filtered, freqs_bpm, spectrum = compute_bpm(signal, fps=fps)
            st.caption(f"Read {len(signal)} frames at ~{fps:.0f} fps")

            if bpm:
                st.session_state.measured_bpm = bpm
                st.metric("Measured pulse", f"{bpm} bpm")

                st.markdown("#### ❤️ Filtered pulse signal (the heartbeat)")
                st.markdown(
                    "Fingertip brightness over time, with slow lighting drift removed. "
                    "Each **peak is one heartbeat** — evenly spaced peaks mean a clean read."
                )
                st.line_chart(filtered)

                st.markdown("#### 📊 Heart-rate frequency spectrum")
                st.markdown(
                    "How strong each possible heart rate is in the signal. The tall peak "
                    f"marks the pulse (**{bpm:.0f} bpm**); smaller bumps are noise/harmonics."
                )
                spectrum_df = pd.DataFrame({"bpm": freqs_bpm, "strength": spectrum})
                st.line_chart(spectrum_df, x="bpm", y="strength")
            else:
                st.warning(
                    "Couldn't detect a clear pulse — hold the fingertip still on the "
                    "lens for a few seconds, or the lighting may have shifted too much."
                )

    # Optional: live camera as an alternative to uploading
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

# =====================================================================
# STEP 3 — Predict (needs all details filled AND a measured pulse)
# =====================================================================
st.header("3️⃣ Predict")

pulse_rate = (
    int(st.session_state.measured_bpm) if st.session_state.measured_bpm else None
)

if not details_done:
    st.info("Complete step 1 first.")
elif not pulse_rate:
    st.info("Measure the pulse in step 2 first.")
else:
    st.write(f"Using measured pulse: **{pulse_rate} bpm**")

    if st.button("Predict risk", type="primary"):
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
