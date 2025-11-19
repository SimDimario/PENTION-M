# service_source_localization_piml.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models/")

# Caricamento modello e scaler PIML
logger.info(f"[PIML] Loading model from {MODEL_PATH}")
model = joblib.load(os.path.join(MODEL_PATH, "emission_source_model_piml.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler_piml.pkl"))
MODEL_VERSION = "RF_PIML_v1"

def estrai_feature(time, conc, window=None, spike_height=None):
    time, conc = np.array(time), np.array(conc)
    if len(time) == 0 or len(conc) == 0:
        return {k: 0.0 for k in ["C_max","t_peak","t_first_peak","mean","std","AUC","rise_rate","fall_rate","plume_duration","spike_count","spike_frequency"]}

    C_max = np.max(conc)
    idx_max = np.argmax(conc)
    t_peak = time[idx_max]
    spike_height = spike_height or 0.1 * C_max
    peaks, _ = find_peaks(conc, height=spike_height)
    t_first_peak = time[peaks[0]] if len(peaks) > 0 else 0.0
    rise_rate = (C_max - conc[0]) / (t_peak - time[0] + 1e-6)
    fall_rate = (C_max - conc[-1]) / (time[-1] - t_peak + 1e-6)
    mean_val, std_val = float(np.mean(conc)), float(np.std(conc))
    auc = np.trapz(conc, time)
    above = np.where(conc > spike_height)[0]
    plume_duration = time[above[-1]] - time[above[0]] if len(above) > 1 else 0.0
    spike_count, total_duration = len(peaks), (time[-1] - time[0] + 1e-6)
    spike_freq = spike_count / total_duration
    return {"C_max":C_max,"t_peak":t_peak,"t_first_peak":t_first_peak,"mean":mean_val,"std":std_val,"AUC":auc,
            "rise_rate":rise_rate,"fall_rate":fall_rate,"plume_duration":plume_duration,
            "spike_count":spike_count,"spike_frequency":spike_freq}

def predict_source_piml(sensors: list, n_sensor_operating: int):
    logger.info(f"[PIML] Predicting source for {len(sensors)} sensors")

    df = pd.DataFrame([s.dict() for s in sensors])
    if df.empty:
        logger.warning("[PIML] No sensor data provided!")
        return {"x": None, "y": None, "confidence": 0.0, "model_version": MODEL_VERSION}

    agg_features = []
    for sensor_id, group in df.groupby("sensor_id"):
        feat = estrai_feature(group["time"], group["conc"])
        first = group.iloc[0]
        feat.update({
            "wind_dir_x": first["wind_dir_x"],
            "wind_dir_y": first["wind_dir_y"],
            "wind_speed": first["wind_speed"],
            "wind_type": first["wind_type"],
            "gps_x": first.get("gps_x", 0.0),
            "gps_y": first.get("gps_y", 0.0),
            "stability_value": first.get("stability_value", 0.0),
            "n_sens_valid": n_sensor_operating,
        })
        agg_features.append(feat)

    X_input = pd.DataFrame(agg_features).fillna(0)
    X_input = X_input.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled = scaler.transform(X_input)
    y_pred = model.predict(X_scaled)

    x, y = float(y_pred[0][0]), float(y_pred[0][1])
    conf = float(np.exp(-np.var(y_pred)))  # semplice proxy di confidenza

    logger.info(f"[PIML] Predicted source: x={x:.3f}, y={y:.3f}, conf={conf:.3f}")
    return {
        "x": x,
        "y": y,
        "confidence": conf,
        "model_version": MODEL_VERSION,
        "predicted_source_xy": [x, y]
    }

