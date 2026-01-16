import numpy as np
import requests
import json

def load_gaussian_map(path):
    arr = np.load(path)
    if arr.ndim == 3:
        return np.mean(arr, axis=2)
    return arr

def request_piml_map(gaussian_map, building_map, wind_speed, wind_dir_list, global_features=None):
    url = "http://localhost:8008/correct_dispersion"

    payload = {
        "wind_speed": float(wind_speed),
        "wind_dir": wind_dir_list,
        "concentration_map": gaussian_map.tolist(),
        "building_map": building_map.tolist(),
        "global_features": global_features if global_features is not None else [],
        "concentration_tensor_3d": None
    }

    res = requests.post(url, json=payload)
    res.raise_for_status()
    data = res.json()
    return np.array(data["corrected_map"]), data["model_version"]

def load_building_map(path):
    return np.load(path)