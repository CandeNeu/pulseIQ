from pydantic import BaseModel


class PatientFeatures(BaseModel):
    age: int
    gender: str
    pulse_rate: int
    systolic_bp: int
    diastolic_bp: int
    glucose: float
    height: float
    weight: float
    bmi: float
    family_diabetes: int
    hypertensive: int
    family_hypertension: int
    cardiovascular_disease: int
    stroke: int
