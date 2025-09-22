from pydantic import BaseModel, Field
from typing import List

class Prediction(BaseModel):
    label: str
    prob: float
