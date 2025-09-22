from pydantic import BaseModel, Field
from typing import List

class Prediction(BaseModel):
    label: str
    prob: float

class AttackResponse(BaseModel):
    clean_topk: List[Prediction]
    adversarial_topk: List[Prediction]
    epsilon: float = Field(..., ge=0.0, le=1.0)
    attack_success: bool
    # adversarial_image_base64_png: str
