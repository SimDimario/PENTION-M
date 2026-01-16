import numpy as np
import joblib
import os
import logging
import json
from scipy.special import softmax
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)
dnn_path = os.path.join(base_dir, "model", "dnn_spectra_version.keras")
brf_path = os.path.join(base_dir, "model", "balanced_random_forest_brf.pkl")
scaler_path = os.path.join(base_dir, "model", "scale_dnn.pkl")

xgb_model_path = os.path.join(base_dir, "model", "xgb_nps_model.json")
xgb_scaler_path = os.path.join(base_dir, "model", "xgb_scaler.pkl")

logger.info("Loading models...")
dnn_clf = load_model(dnn_path)
logger.info("DNN loaded")
brf_clf = joblib.load(brf_path)
logger.info("BRF loaded")
scaler_dnn = joblib.load(scaler_path)
logger.info("Scaler DNN loaded")

xgb_clf = XGBClassifier()
xgb_clf.load_model(xgb_model_path)
logger.info("XGB loaded")

xgb_scaler = joblib.load(xgb_scaler_path)
logger.info("Scaler XGB loaded")

mz_range = np.arange(1, 601)

legends = {
    0: "Cathinone analogues",
    1: "Cannabinoid analogues",
    2: "Phenethylamine analogues",
    3: "Piperazine analogues",
    4: "Tryptamine analogues",
    5: "Fentanyl analogues",
    6: "Other compounds",
}


def _compute_features(spectrum):
    """Extracts 13 features from the mass spectrum."""
    logger.debug("Spectrum feature calculation")

    peaks = [
        (mz, intensity) for mz, intensity in zip(mz_range, spectrum) if intensity > 0
    ]

    if not peaks:
        logger.warning("Spectrum without peaks")
        return [np.nan] * 13

    mz_values, intensities = zip(*peaks)
    mz_values = np.array(mz_values)
    intensities = np.array(intensities)

    base_peak_idx = np.argmax(intensities)
    base_peak_mass = mz_values[base_peak_idx]

    base_prox = (
        np.min(
            np.abs(mz_values - base_peak_mass)[np.abs(mz_values - base_peak_mass) != 0]
        )
        if len(mz_values) > 1
        else 0.0
    )
    max_mass = np.max(mz_values)
    max_prox = (
        np.min(np.abs(mz_values - max_mass)[np.abs(mz_values - max_mass) != 0])
        if len(mz_values) > 1
        else 0.0
    )
    num_peaks = len(peaks)
    intensity_mean = np.mean(intensities)
    intensity_std = np.std(intensities)
    intensity_density = np.max(intensities) / num_peaks
    mass_mean = np.mean(mz_values)
    mass_std = np.std(mz_values)
    mass_density = max_mass / num_peaks

    diffs = np.abs(np.subtract.outer(mz_values, mz_values))
    diffs = diffs[np.triu_indices(len(diffs), k=1)]
    diff_counts = np.bincount(np.round(diffs).astype(int))
    ppmd = np.argmax(diff_counts) if len(diff_counts) > 0 else 0
    mean_ppmd = np.mean(diffs) if len(diffs) > 0 else 0

    return [
        base_peak_mass,
        base_prox,
        max_mass,
        max_prox,
        num_peaks,
        intensity_mean,
        intensity_std,
        intensity_density,
        mass_mean,
        mass_std,
        mass_density,
        ppmd,
        mean_ppmd,
    ]


def pipe_clf_dnn(spectra: np.ndarray, T: float = 2.5):
    """
    DNN + Temperature Scaling for realistic confidence.
    """
    if spectra is None or len(spectra) == 0:
        raise ValueError("Input spectra is empty or None")
    if spectra.ndim != 2:
        raise ValueError(
            f"Expected 2D array (n_samples, n_features), got shape {spectra.shape}"
        )

    try:
        logger.info("Inizio predizione DNN")
        spectra_scaled = scaler_dnn.transform(spectra)

        probs_raw = dnn_clf.predict(spectra_scaled, verbose=0)[0]

        probs_raw = np.clip(probs_raw, 1e-9, 1.0)

        logits = np.log(probs_raw)

        scaled_logits = logits / T
        scaled_probs = softmax(scaled_logits)

        pred_idx = int(np.argmax(scaled_probs))
        confidence = float(np.max(scaled_probs))

        predictions = [legends.get(pred_idx, f"Class {pred_idx}")]
        logger.info("DNN prediction completed (Temperature Scaling)")

        return {"predictions": predictions, "confidence": confidence}

    except Exception as e:
        logger.exception("Errore durante la predizione DNN")
        raise RuntimeError(f"Error during DNN prediction: {str(e)}")


def pipe_clf_brf(spectra: np.ndarray):
    if spectra is None or len(spectra) == 0:
        raise ValueError("Input spectra is empty or None")
    if spectra.ndim != 2:
        raise ValueError(
            f"Expected 2D array (n_samples, n_features), got shape {spectra.shape}"
        )

    predictions = []
    try:
        logger.info("BRF Prediction Start")
        for idx, spectrum in enumerate(spectra):
            logger.debug(f"Spectrum feature calculation {idx}")
            features = np.array(_compute_features(spectrum))
            selected_indices = [0, 1, 3, 5, 6, 7, 8, 10, 11]
            features = features[selected_indices].reshape(1, -1)
            prediction = brf_clf.predict(features)
            predictions.append(legends.get(prediction[0], f"Class {prediction[0]}"))
        logger.info("BRF prediction completed")
    except Exception as e:
        logger.exception("Error during BRF prediction")
        raise RuntimeError(f"Error during BRF prediction: {str(e)}")

    return np.array(predictions)


def load_temperature():
    config_path = os.path.join(base_dir, "model", "temp_config.json")
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return float(cfg.get("T", 1.8))
    except:
        return 1.8


def pipe_clf_xgb(spectra: np.ndarray, dynamic_T: float | None = None):
    if spectra is None or len(spectra) == 0:
        raise ValueError("Input spectra is empty or None")
    if spectra.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, 600), got {spectra.shape}")

    try:
        spectra_scaled = xgb_scaler.transform(spectra)

        raw_probs = xgb_clf.predict_proba(spectra_scaled)[0]

        if dynamic_T is not None:
            T = float(dynamic_T)
        else:
            T = load_temperature()

        logits = np.log(np.clip(raw_probs, 1e-12, 1))
        logits_scaled = logits / T
        probs = softmax(logits_scaled)

        pred_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        prediction = legends.get(pred_idx, f"Class {pred_idx}")

        return {
            "predictions": [prediction],
            "confidence": confidence,
            "temperature_used": T,
            "model": "XGB_v2",
        }

    except Exception as e:
        raise RuntimeError(f"Error during XGB prediction: {str(e)}")
