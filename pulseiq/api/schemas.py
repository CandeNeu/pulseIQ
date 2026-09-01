from pydantic import BaseModel


class PatientFeatures(BaseModel):
    sex: int
    age: int
    height: float
    weight: float
    bmi: float
    pulse: float
    ever_smoked: int
    current_smoker: int
