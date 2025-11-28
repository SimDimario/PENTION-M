from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
import json
import os
import sys

# Percorso locale corretto
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from service_correction_piml import correct_dispersion_piml
from binary_map_gen import generate_binary_map, convert_np
import uvicorn

# === CARICAMENTO BINARY MAP DI DEFAULT (per allineare il modello) ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data", "amsterdam_netherlands_bbox.npy")

if os.path.exists(DEFAULT_MAP_PATH):
    DEFAULT_BUILDING_MAP = np.load(DEFAULT_MAP_PATH)
    print(f"[INFO] Loaded default building map: {DEFAULT_BUILDING_MAP.shape}")
else:
    print("[WARN] Default binary map not found — using empty map.")
    DEFAULT_BUILDING_MAP = np.zeros((500, 500), dtype=np.float32)

app = FastAPI()

class BBox(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    grid_size: int = 300
    place: str = "Amsterdam, Netherlands"

class DispersionInput(BaseModel):
    wind_speed: float
    wind_dir: list
    concentration_map: list
    building_map: list
    global_features: list | None = None
    concentration_tensor_3d: Optional[list] = None

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

    build_map = (
        np.array(payload.building_map, dtype=np.float32)
        if len(payload.building_map) > 0
        else DEFAULT_BUILDING_MAP
    )

    glob_feat = None  # le global features non sono usate nella versione attuale del modello

    C_tensor = None
    if payload.concentration_tensor_3d is not None:
        try:
            C_tensor = np.array(payload.concentration_tensor_3d, dtype=np.float32)
            if C_tensor.ndim != 3:
                print(f"[WARN] concentration_tensor_3d shape inattesa: {C_tensor.shape}")
                C_tensor = None
            else:
                print(f"[INFO] Tensor 3D ricevuto dal ingestion: shape={C_tensor.shape}")
        except Exception as e:
            print(f"[WARN] Errore nel parsing del tensor 3D: {e}")
            C_tensor = None

    correction_result = correct_dispersion_piml(
        payload.wind_dir,
        payload.wind_speed,
        conc_map,
        build_map,
        glob_feat,
        C_tensor
    )

    return {
        "status": "ok",
        **correction_result   # contiene corrected_map + model_version
    }

"""
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)"""