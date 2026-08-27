from pydantic import BaseModel


class PatientFeatures(BaseModel):
    age: int

    pulse_rate: int



    height: float
    weight: float
    bmi: float
