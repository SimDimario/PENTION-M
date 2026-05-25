from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime


class Coordinates(BaseModel):

    latitude: float

    longitude: float


class SamplingPoint(BaseModel):

    id: str

    coordinates: Coordinates

    measured_concentrations: Dict[str, float]

    sample_time: datetime


class Units(BaseModel):

    concentration: str

    coordinates: str


class TrajectoryRequest(BaseModel):

    sampling_points: List[SamplingPoint]

    units: Units