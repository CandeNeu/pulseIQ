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

# ============ Grundinställningar och Custom CSS ============
st.set_page_config(page_title="PulseIQ", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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

API_URL = os.environ.get(
    "API_URL", "https://pulseiq-api-431111687933.europe-west1.run.app/predict"
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
    """Körs på varje kamerabild – extraherar röd ljusstyrka (PPG-signal)."""
    img = frame.to_ndarray(format="bgr24")
    with lock:
        signal_buffer.append(float(img[:, :, 2].mean()))
    return frame


def compute_bpm(signal, fps=30.0):
    """Uppskattar puls från en ljusstyrkesignal via bandpass + FFT.

    Returnerar (bpm, filtered_signal, freqs_bpm, spectrum).
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
    """Läser upp till max_frames av röd ljusstyrka från mitten av varje bild.

    Returnerar (signal, fps). imageio ger RGB, så röd är index 0.
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
        crop = frame[cy - 50 : cy + 50, cx - 50 : cx + 50, 0]  # mitten, röd kanal
        signal.append(float(crop.mean()))
        if len(signal) >= max_frames:
            break
    return signal, fps


# ============ Huvudlayout med flikar ============
tab1, tab2, tab3 = st.tabs(
    ["📝 1. Patientdetaljer", "❤️ 2. Pulsmätning", "📊 3. Analys & Resultat"]
)

# ==========================================
# FLIK 1: PATIENTDETALJER
# ==========================================
with tab1:
    st.markdown("### Fyll i patientinformation")

    col_demo, col_vitals, col_history = st.columns(3)

    with col_demo:
        st.markdown("**Demografi & Kroppsmått**")
        age = st.number_input("Ålder", min_value=0, max_value=120, value=33)
        gender = st.selectbox("Kön", ["Male", "Female"])
        height = st.number_input("Längd (m)", min_value=1.0, max_value=2.5, value=1.63)
        weight = st.number_input(
            "Vikt (kg)", min_value=20.0, max_value=250.0, value=70.2
        )
        bmi = round(weight / (height**2), 2) if height > 0 else 0.0
        st.metric(label="Beräknat BMI", value=bmi)

    with col_vitals:
        st.markdown("**Vitalparametrar**")
        systolic_bp = st.number_input("Systoliskt BT", value=106)
        diastolic_bp = st.number_input("Diastoliskt BT", value=67)
        glucose = st.number_input("Glukos", value=5.81)

    with col_history:
        st.markdown("**Medicinsk Historik**")
        family_diabetes = st.selectbox("Familjehistorik av diabetes", ["No", "Yes"])
        hypertensive = st.selectbox("Hypertoni", ["No", "Yes"])
        family_hypertension = st.selectbox(
            "Familjehistorik av hypertoni", ["No", "Yes"]
        )
        cardiovascular = st.selectbox("Kardiovaskulär sjukdom", ["No", "Yes"])
        stroke = st.selectbox("Stroke", ["No", "Yes"])

# ==========================================
# FLIK 2: PULSMÄTNING
# ==========================================
with tab2:
    st.markdown("### Ladda upp video för pulsmätning")

    col_upload, col_metric = st.columns([2, 1])

    with col_upload:
        video = st.file_uploader(
            "Upload a fingertip video",
            type=["mp4", "mov", "avi", "webm"],
            label_visibility="collapsed",
        )

    if video:
        with st.spinner("Analyserar video..."):
            signal, fps = extract_signal_from_video(video.getvalue())

        if len(signal) < 10:
            st.warning("Videon verkar för kort för att analyseras.")
        else:
            bpm, filtered, freqs_bpm, spectrum = compute_bpm(signal, fps=fps)

            if bpm:
                st.session_state.measured_bpm = bpm
                with col_metric:
                    st.metric(label="Uppmätt Puls", value=f"{bpm} bpm")

                graph_col1, graph_col2 = st.columns(2)
                with graph_col1:
                    st.markdown("#### ❤️ Filtrerad pulssignal")
                    st.caption(
                        "Fingerspetsens ljusstyrka över tid, med långsam drift "
                        "borttagen. Varje topp är ett hjärtslag."
                    )
                    st.line_chart(filtered)
                with graph_col2:
                    st.markdown("#### 📊 Frekvensspektrum")
                    st.caption(
                        f"Visar hur stark varje möjlig puls är i signalen. "
                        f"Den höga toppen markerar pulsen ({bpm:.0f} bpm)."
                    )
                    spectrum_df = pd.DataFrame({"bpm": freqs_bpm, "strength": spectrum})
                    st.line_chart(spectrum_df, x="bpm", y="strength")
            else:
                st.warning(
                    "Kunde inte hitta en tydlig puls — håll fingerspetsen stilla "
                    "mot linsen i några sekunder."
                )

    with st.expander("📹 Or measure from the live camera instead"):
        st.caption("Täck linsen med fingerspetsen och håll stilla.")
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
                        bpm_ph.metric("Uppmätt Puls", f"{bpm} bpm")
                time.sleep(0.3)

# ==========================================
# FLIK 3: PREDIKTION
# ==========================================
with tab3:
    st.markdown("### Starta Riskbedömning")

    pulse_rate = (
        int(st.session_state.measured_bpm)
        if st.session_state.get("measured_bpm")
        else None
    )

    if pulse_rate:
        st.info(
            f"**Redo för analys:** BMI är {bmi} och uppmätt puls är {pulse_rate} bpm."
        )
    else:
        st.warning("Mät pulsen i flik 2 innan du gör en analys.")

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

        with st.spinner("Analyserar via API..."):
            try:
                response = requests.get(API_URL, params=params)

                if response.status_code != 200:
                    st.error(f"API returnerade status {response.status_code}")
                    st.code(response.text)
                    st.stop()

                try:
                    result = response.json()
                except requests.exceptions.JSONDecodeError:
                    st.error("API:et returnerade inte giltig JSON. Rått svar nedan:")
                    st.code(response.text)
                    st.stop()

                st.success("Riskbedömning slutförd!")

                res_col1, res_col2 = st.columns(2)

                risk = result.get("diabetic_risk")
                if risk is not None:
                    res_col1.metric("Diabetes Risk", f"{risk:.1%}")

                pred = result.get("diabetic_prediction")
                if pred is not None:
                    label = (
                        "Diabetic"
                        if str(pred).lower() in ("yes", "1")
                        else "Not Diabetic"
                    )
                    res_col2.metric("Prediktion", label)

                with st.expander("Rått API-svar"):
                    st.write(result)

            except requests.exceptions.RequestException as e:
                st.error(f"Kunde inte nå API:et: {e}")
