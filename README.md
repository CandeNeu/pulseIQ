# 🩺 PulseIQ – Diabetes Risk Prediction

A machine learning web application that predicts diabetes risk from patient
health data. Built with a **scikit-learn** model served through a **FastAPI**
backend and a **Streamlit** frontend.

The model is trained on the *DiaBD – A Diabetes Dataset for Enhanced Risk
Analysis and Research in Bangladesh* and reaches **~94.5% accuracy** on the
held-out test set.

---

## 🏗️ Architecture


The frontend collects 14 patient features, sends them to the API, which runs
the trained pipeline and returns a prediction plus an estimated risk score.

---

## 📁 Project structure


---

## 🧬 Features used by the model

| Feature | Description |
|---|---|
| `age` | Patient age |
| `gender` | Female / Male |
| `pulse_rate` | Resting pulse (bpm) |
| `systolic_bp` | Systolic blood pressure |
| `diastolic_bp` | Diastolic blood pressure |
| `glucose` | Blood glucose level |
| `height` | Height (m) |
| `weight` | Weight (kg) |
| `bmi` | Body Mass Index (auto-calculated) |
| `family_diabetes` | Family history of diabetes (0/1) |
| `hypertensive` | Hypertensive (0/1) |
| `family_hypertension` | Family history of hypertension (0/1) |
| `cardiovascular_disease` | Cardiovascular disease (0/1) |
| `stroke` | History of stroke (0/1) |

**Target:** `diabetic` (Yes / No)

---

## 🚀 Getting started (local)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/pulseIQ.git
cd pulseIQ
```

### 2. Set up the environment

This project uses `pyenv` with a virtual environment named `pulse_env`:

```bash
pyenv virtualenv <your-python-version> pulse_env
pyenv local pulse_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

This creates `models/model.joblib`:

```bash
python pulseiq/ml_logic/train.py
```

You should see the test accuracy and a `✅ Model saved` confirmation.

### 5. Run the API

In one terminal:

```bash
uvicorn pulseiq.api.fast:app --reload --port 8000
```

Check it's running at [http://localhost:8000](http://localhost:8000) — you
should see `{"status":"ok","message":"PulseIQ API is running"}`.

### 6. Run the frontend

In a second terminal:

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL, fill in the patient details, and click **Predict risk**.

---

## 🔌 API reference

### `GET /`
Health check. Returns API status.

### `GET /predict`
Returns a diabetes prediction.

**Query parameters:** all 14 features listed above.

**Example response:**
```json
{
  "diabetic_prediction": "No",
  "diabetic_risk": 0.005
}
```

---

## ☁️ Deployment

The frontend reads the API address from the `API_URL` environment variable,
defaulting to `http://localhost:8000/predict` locally. In production, set
`API_URL` to your deployed API's URL.

<!-- Fill in once deployed -->
- **API:** `<your-cloud-run-api-url>`
- **Frontend:** `<your-frontend-url>`

---

## 🛠️ Tech stack

- **ML:** scikit-learn, pandas, joblib
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit, Requests
- **Deployment:** Google Cloud Run / Streamlit Cloud

---

## 📊 Dataset

*DiaBD – A Diabetes Dataset for Enhanced Risk Analysis and Research in
Bangladesh.*

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and is **not** a medical
device. Predictions must not be used for actual diagnosis or treatment
decisions. Always consult a qualified healthcare professional.
