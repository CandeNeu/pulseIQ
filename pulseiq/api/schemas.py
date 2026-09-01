from pydantic import BaseModel


class PatientFeatures(BaseModel):
    sex: int
    age: int
    weight: float
    height: float
    bmi: float
    pulse: float
    ever_smoked: int
    current_smoker: int
