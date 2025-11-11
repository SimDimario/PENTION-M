from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging
import os
import sys
import numpy as np

# percorso locale del servizio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from service_source_localization_piml import predict_source_piml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class SensorData(BaseModel):
    sensor_id: int
    sensor_is_fault: bool
    time: float | None
    conc: float | None
    wind_dir_x: float | None
    wind_dir_y: float | None
    wind_speed: float | None
    wind_type: int | None
    gps_x: float | None = 0.0
    gps_y: float | None = 0.0
    stability_value: float | None = 0.0

class PredictRequest(BaseModel):
    payload_sensors: List[SensorData]
    n_sensor_operating: int

app = FastAPI(title="EmissionSourceLocalization_PIML")

@app.post("/predict_source_piml")
def predict_source_piml_endpoint(request: PredictRequest):
    logger.info(f"Received /predict_source_piml with {len(request.payload_sensors)} records")

    try:
        result = predict_source_piml(request.payload_sensors, request.n_sensor_operating)
        return {"status": 200, **result}

    except Exception as e:
        logger.exception("Error during PIML source localization prediction")
        raise e

"""if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)"""
