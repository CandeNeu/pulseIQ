import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Ladda data ---
df = pd.read_csv("raw_data/DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv")

# --- 2. Dela features / target ---
X = df.drop(columns=["diabetic"])
y = df["diabetic"]

# --- 3. Definiera kolumntyper ---
categorical = ["gender"]
numeric = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric),
])

# --- 4. Bygg pipeline: preprocessing + modell i ETT objekt ---
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
])

# --- 5. Träna ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# --- 6. Utvärdera ---
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {acc:.3f}")

# --- 7. Spara hela pipelinen ---
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.joblib")
print("✅ Model saved to models/model.joblib")
