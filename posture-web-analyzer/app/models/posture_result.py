from pydantic import BaseModel
from typing import Optional

class NeckResult(BaseModel):
    angle: float
    grade: str
    feedback: str

class SpineResult(BaseModel):
    curvature: float
    feedback: str

class ShoulderResult(BaseModel):
    asymmetry: float
    feedback: str

class PelvicResult(BaseModel):
    tilt: float
    feedback: str

class SpineTwistingResult(BaseModel):
    twisting: float
    feedback: str

class PostureResult(BaseModel):
    detected: bool
    neck: Optional[NeckResult] = None
    spine: Optional[SpineResult] = None
    shoulder: Optional[ShoulderResult] = None
    pelvic: Optional[PelvicResult] = None
    spine_twisting: Optional[SpineTwistingResult] = None
    message: Optional[str] = None