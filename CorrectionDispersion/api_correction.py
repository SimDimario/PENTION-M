from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from CorrectionDispersion.binary_map_gen import generate_binary_map, convert_np
from CorrectionDispersion.service_correction import correct_dispersion
import uvicorn

app = FastAPI()

class BBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    grid_size: int = 300
    place: str = "Roma, Italy"

class DispersionInput(BaseModel):
    wind_speed: float
    wind_dir: list
    concentration_map: list
    building_map: list
    global_features: list | None = None

@app.post("/generate_binary_map")
def generate_map(bbox: BBox):

    quartiere_bbox = (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)

    binary_map, metadata = generate_binary_map(
        place=bbox.place,
        bbox=quartiere_bbox,
        grid_size=bbox.grid_size
    )

    out_dir = "binary_maps_data"
    os.makedirs(out_dir, exist_ok=True)

    """map_filename = os.path.join(".", "CorrectionDispersion/binary_maps_data", f"{bbox.place.lower().replace(', ', '_').replace(' ', '_')}{'_bbox' if quartiere_bbox is not None else ''}.npy")
    meta_filename = os.path.join(".", "CorrectionDispersion/binary_maps_data", f"{bbox.place.lower().replace(', ', '_').replace(' ', '_')}_metadata{'_bbox' if quartiere_bbox is not None else ''}.json")
    
    np.save(map_filename, binary_map)
    with open(meta_filename, "w") as f:
        json.dump(convert_np(metadata), f, indent=4)"""

    return {
        "status_code": "success",
        "map": binary_map.tolist(),
        "metadata": convert_np(metadata)
    }

@app.post("/correct_dispersion")
def predict_endpoint(payload: DispersionInput):

    conc_map = np.array(payload.concentration_map, dtype=np.float32)
    build_map = np.array(payload.building_map, dtype=np.float32)
    glob_feat = np.array(payload.global_features, dtype=np.float32) if payload.global_features else None

    correction_map = correct_dispersion(payload.wind_dir, payload.wind_speed, conc_map, build_map, glob_feat)

    return {"status_code": "success",
            "predictions": correction_map.tolist()}

"""
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)"""