import torch
import numpy as np
from MCxM_PIML import MCxM_PIML
import os
import logging
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "mcxm_piml_model_best.pth")
BINARY_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data", "amsterdam_netherlands_bbox.npy")
MODEL_REGISTRY_PATH = "/logs/model_registry.json"
CACHED_MODEL = None
CACHED_VERSION = None
CACHED_DEVICE = None
CACHED_BINARY_MAP = None

REGISTRY_PATH = "/logs/model_registry.json"

def get_model_version():
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("current_model_version", "PIML_v1")
        except:
            return "PIML_v1"
    return "PIML_v1"

logger = logging.getLogger("CorrectionDispersion")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
def calculate_mean_direction(wind_dir_array):
    wind_dir_rad = np.radians(wind_dir_array)
    cos_vals = np.cos(wind_dir_rad)
    sin_vals = np.sin(wind_dir_rad)
    mean_cos = np.mean(cos_vals)
    mean_sin = np.mean(sin_vals)
    return mean_cos, mean_sin

def load_model_if_needed():
    """
    Carica il modello MCxM_PIML se:
    - non è mai stato caricato
    - oppure è cambiata la versione nel model_registry.json
    """
    global CACHED_MODEL, CACHED_VERSION, CACHED_DEVICE, CACHED_BINARY_MAP
    new_version = None
    if os.path.exists(MODEL_REGISTRY_PATH):
        try:
            with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            new_version = data.get("current_model_version")
        except Exception as e:
            print(f"[PIML] [WARN] Impossibile leggere model_registry: {e}")
            new_version = None

    reload_needed = False

    if CACHED_MODEL is None:
        reload_needed = True
    elif new_version is not None and CACHED_VERSION != new_version:
        reload_needed = True

    if not reload_needed:
        return CACHED_MODEL, CACHED_VERSION, CACHED_DEVICE, CACHED_BINARY_MAP

    if not os.path.exists(BINARY_MAP_PATH):
        raise FileNotFoundError(f"Binary map non trovata: {BINARY_MAP_PATH}")
    binary_map = np.load(BINARY_MAP_PATH)
    m = binary_map.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MCxM_PIML(binary_map, m=m, n_channel=1, wind_dim=2, n_global_features=0).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modello PIML non trovato: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    CACHED_MODEL = model
    CACHED_VERSION = new_version or "unknown"
    CACHED_DEVICE = device
    CACHED_BINARY_MAP = binary_map

    print(f"[PIML] Modello PIML caricato/ricaricato. Versione: {CACHED_VERSION}, device={device}")

    return CACHED_MODEL, CACHED_VERSION, CACHED_DEVICE, CACHED_BINARY_MAP

def correct_dispersion_piml(
    wind_dir,
    wind_speed,
    concentration_map,
    building_map,
    global_feature=None,
    C_tensor=None,
    device=None,
    m=500,
    pretrained_path=None
):
    logger.info("Starting dispersion correction...")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f"Using device: {device}")

    binary_map_path = os.path.join(
        SCRIPT_DIR,
        "binary_maps_data",
        "amsterdam_netherlands_bbox.npy"
    )

    if building_map is None:
        logger.warning("[WARN] No building_map provided — loading default binary map")
        if os.path.exists(binary_map_path):
            building_map = np.load(binary_map_path)
            logger.info(f"Loaded binary map from {binary_map_path} with shape {building_map.shape}")
        else:
            logger.error(f"[ERROR] Default binary map not found at {binary_map_path}")
            building_map = np.zeros((m, m), dtype=np.float32)
    else:
        logger.info(f"Received building_map with shape {building_map.shape}")

    logger.info(f"building map: {building_map.shape}")

    model, cached_version, device, _ = load_model_if_needed()
    logger.info(f"[PIML] Using cached/loaded model version: {cached_version}")

    if concentration_map is None or concentration_map.size == 0:
        logger.warning("Empty concentration_map received — using zeros fallback.")
        concentration_map = np.zeros((m, m, 1), dtype=np.float32)
    elif concentration_map.ndim == 1:
        logger.warning(f"1D concentration_map of shape {concentration_map.shape} — reshaping to (m, m, 1).")
        concentration_map = concentration_map.reshape(m, m, 1)
    elif concentration_map.ndim == 2:
        logger.warning(f"2D concentration_map of shape {concentration_map.shape} — adding fake time axis.")
        concentration_map = concentration_map[:, :, np.newaxis]

    p95 = np.percentile(concentration_map, 95)
    concentration_map = concentration_map / (p95 + 1e-6)
    concentration_map = np.clip(concentration_map, 0, 1)

    cm_agg = np.mean(concentration_map, axis=2)
    mc = torch.tensor(cm_agg, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    logger.debug(f"Concentration map tensor shape: {mc.shape}")

    wind_dir_cos, wind_dir_sin = calculate_mean_direction(wind_dir)
    degree_angle = np.degrees(np.arctan2(wind_dir_sin, wind_dir_cos)) % 360
    logger.debug(f"Wind direction: {degree_angle}°")

    wind_features = torch.tensor([[wind_speed, degree_angle]], dtype=torch.float32, device=device)
    logger.debug(f"Wind features tensor: {wind_features}")

    global_features = None

    logger.info("Running model inference...")
    with torch.no_grad():
        try:
            output = model(mc, wind_features)
            output = output.detach().cpu().numpy()[0]
            logger.info(f"Inference completed. Output shape: {output.shape}")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise e

    return {
        "corrected_map": output.tolist(),
        "model_version": get_model_version()
    }