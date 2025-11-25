import torch
import numpy as np
from MCxM_PIML import MCxM_PIML
import os
import logging
import json

# === GLOBAL MODEL CACHE ===
GLOBAL_MODEL = None
GLOBAL_MAP = None

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def calculate_mean_direction(wind_dir_array):
    # Convert to radians
    wind_dir_rad = np.radians(wind_dir_array)

    # Calculate directional vectors
    cos_vals = np.cos(wind_dir_rad)
    sin_vals = np.sin(wind_dir_rad)

    # Calculate mean
    mean_cos = np.mean(cos_vals)
    mean_sin = np.mean(sin_vals)

    return mean_cos, mean_sin

def load_model(binary_map, m=500, device=None, pretrained_path=None):
    global GLOBAL_MODEL, GLOBAL_MAP

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = os.path.join(SCRIPT_DIR, "models")
    best_path = os.path.join(model_dir, "mcxm_piml_model_best.pth")
    final_path = os.path.join(model_dir, "mcxm_piml_model_final.pth")

    # === MODEL CACHING: se il modello è già caricato e la mappa è la stessa, riusa ===
    if GLOBAL_MODEL is not None and GLOBAL_MAP is not None:
        if np.array_equal(GLOBAL_MAP, binary_map):
            logger.info("Using cached PIML model (no reload).")
            return GLOBAL_MODEL

    logger.info(f"Initializing MCxM_PIML model (device={device})")
    loaded_model = MCxM_PIML(
        binary_map,
        m=m,
        n_channel=1,
        wind_dim=2,
        n_global_features=0
    ).to(device)

    # === Caso: path esplicito ===
    if pretrained_path is not None:
        ckpt_path = pretrained_path
        logger.info(f"Loading PIML model from custom path: {ckpt_path}")
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            loaded_model.load_state_dict(state_dict)
            loaded_model.eval()
            GLOBAL_MODEL = loaded_model
            GLOBAL_MAP = binary_map
            logger.info("Custom model loaded and cached.")
            return loaded_model
        except Exception as e:
            logger.error(f"Error loading custom model from {ckpt_path}: {e}")
            raise

    # === 1) Provare a caricare il BEST ===
    if os.path.exists(best_path):
        logger.info(f"Trying to load BEST model from {best_path}")
        try:
            data = torch.load(best_path, map_location=device)

            # Caso A — {"state_dict": ...}
            if isinstance(data, dict) and "state_dict" in data:
                state_dict = data["state_dict"]

            # Caso B — state_dict puro
            elif isinstance(data, dict):
                state_dict = data

            # Caso C — file salvato male (torch.save(model, ...))
            else:
                raise RuntimeError(
                    "BEST model is not a valid state_dict — re-export required."
                )

            loaded_model.load_state_dict(state_dict)
            loaded_model.eval()

            # Salvataggio in cache
            GLOBAL_MODEL = loaded_model
            GLOBAL_MAP = binary_map

            logger.info("BEST model loaded, validated and cached.")
            return loaded_model

        except Exception as e:
            logger.error(f"Error loading BEST model: {e}")

    # === 2) Fallback sul FINAL ===
    if os.path.exists(final_path):
        logger.info(f"Trying to load FINAL model from {final_path}")
        try:
            state_dict = torch.load(final_path, map_location=device)
            loaded_model.load_state_dict(state_dict)
            loaded_model.eval()

            GLOBAL_MODEL = loaded_model
            GLOBAL_MAP = binary_map

            logger.info("FINAL model loaded and cached.")
            return loaded_model
        except Exception as e:
            logger.error(f"Error loading FINAL model: {e}")

    # === 3) Nessun modello valido ===
    err_msg = (
        "No valid PIML model could be loaded. "
        f"Tried BEST ({best_path}) and FINAL ({final_path})."
    )
    logger.critical(err_msg)
    raise RuntimeError(err_msg)

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

    model = load_model(building_map, m=m, device=device, pretrained_path=pretrained_path)

    # --- shape concentration_map ------------------------------------------
    if concentration_map is None or concentration_map.size == 0:
        logger.warning("Empty concentration_map received — using zeros fallback.")
        concentration_map = np.zeros((m, m, 1), dtype=np.float32)
    elif concentration_map.ndim == 1:
        logger.warning(f"1D concentration_map of shape {concentration_map.shape} — reshaping to (m, m, 1).")
        concentration_map = concentration_map.reshape(m, m, 1)
    elif concentration_map.ndim == 2:
        logger.warning(f"2D concentration_map of shape {concentration_map.shape} — adding fake time axis.")
        concentration_map = concentration_map[:, :, np.newaxis]

    # normalizzazione runtime (p95, coerente col training)
    p95 = np.percentile(concentration_map, 95)
    concentration_map = concentration_map / (p95 + 1e-6)
    concentration_map = np.clip(concentration_map, 0, 1)

    cm_agg = np.mean(concentration_map, axis=2)
    mc = torch.tensor(cm_agg, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    logger.debug(f"Concentration map tensor shape: {mc.shape}")

    # vento medio
    wind_dir_cos, wind_dir_sin = calculate_mean_direction(wind_dir)
    degree_angle = np.degrees(np.arctan2(wind_dir_sin, wind_dir_cos)) % 360
    logger.debug(f"Wind direction: {degree_angle}°")

    wind_features = torch.tensor([[wind_speed, degree_angle]], dtype=torch.float32, device=device)
    logger.debug(f"Wind features tensor: {wind_features}")

    global_features = None  # non usate in questa versione

    logger.info("Running model inference...")
    with torch.no_grad():
        try:
            output = model(mc, wind_features)          # [1,m,m]
            output = output.detach().cpu().numpy()[0]  # (m,m)  <-- NIENTE * building_map
            logger.info(f"Inference completed. Output shape: {output.shape}")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise e

    return {
        "corrected_map": output.tolist(),
        "model_version": get_model_version()
    }
