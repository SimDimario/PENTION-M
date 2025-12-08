from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
import json
import sys
import numpy as np
import pandas as pd
import logging

# ============================================================
# LOGGER
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# FIX PATH PRIMA DELL’IMPORT
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ClassificatoreNPS import service_clf_nps


# ============================================================
# MODELLI INPUT
# ============================================================

class Spectra(BaseModel):
    spectra: list[list[float]]
    def to_numpy(self) -> np.ndarray:
        return np.array(self.spectra, dtype=float)

class GenRequest(BaseModel):
    noise_level: float = 0.05
    concentration: float = 0.5
    compound_name: str | None = None


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="NPS Classifier Service")

# ============================================================
# ENDPOINT UFFICIALI USATI DA INGESTION
# ============================================================

@app.post("/predict_dnn")
def predict_dnn(input_data: Spectra):
    logger.info("Richiesta su /predict_dnn → redirect to XGB")
    try:
        mass_spectrum = input_data.to_numpy()
        result = service_clf_nps.pipe_clf_xgb(mass_spectrum)

        return JSONResponse(
            content={
                "predictions": result["predictions"],
                "confidence": result["confidence"],
                "model": "XGB"
            },
            status_code=200
        )
    except Exception as e:
        logger.exception("Errore in /predict_dnn")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict_brf")
def predict_brf(input_data: Spectra):
    logger.info("Richiesta su /predict_brf")
    try:
        mass_spectrum = input_data.to_numpy()
        predictions = service_clf_nps.pipe_clf_brf(mass_spectrum)

        return JSONResponse(
            content={"predictions": predictions.tolist()},
            status_code=200
        )
    except Exception as e:
        logger.exception("Errore in /predict_brf")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict_xgb")
def predict_xgb(input_data: Spectra, dynamic_T: float | None = None):
    logger.info("Richiesta su /predict_xgb")
    try:
        mass_spectrum = input_data.to_numpy()
        result = service_clf_nps.pipe_clf_xgb(mass_spectrum, dynamic_T)

        return JSONResponse(
            content={
                "predictions": result["predictions"],
                "confidence": result["confidence"],
                "temperature_used": result.get("temperature_used", None),
                "model": "XGB"
            },
            status_code=200
        )

    except Exception as e:
        logger.exception("Errore in /predict_xgb")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/set_temperature/{T}")
def set_temperature(T: float):
    config_path = os.path.join(os.path.dirname(__file__), "model", "temp_config.json")
    try:
        with open(config_path, "w") as f:
            json.dump({"T": T}, f)
        return {"status": "ok", "new_temperature": T}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ============================================================
# ENDPOINT DI DEBUG (NON USATO DA INGESTION)
# ============================================================

@app.post("/generate_and_predict")
def generate_and_predict(req: GenRequest):

    try:
        dataset_path = "/ClassificatoreNPS/datasetNPS/PENTION_EI_Complete.csv"
        df = pd.read_csv(dataset_path)

        # selezione sostanza
        if req.compound_name is None or req.compound_name.strip().lower() == "unknown":
            row = df.sample(n=1).iloc[0]
        else:
            row = df[df["Name"].str.lower() == req.compound_name.lower()]
            row = row.iloc[0] if not row.empty else df.sample(n=1).iloc[0]

        true_name = row["Name"]

        # spettro reale 1–600
        spectrum = row.iloc[1:601].values.astype(float)

        # rumore realistico
        noise = np.random.normal(
            0,
            req.noise_level * 5.0,
            spectrum.shape
        )
        spectrum_noisy = np.clip(spectrum + noise, 0, None)

        # classificazione
        res = service_clf_nps.pipe_clf_dnn(np.array([spectrum_noisy]))

        return JSONResponse(
            content={
                "true_compound": true_name,
                "predicted_class": res["predictions"][0],
                "confidence": res["confidence"],
                "spectrum_noisy": spectrum_noisy.tolist()
            },
            status_code=200
        )

    except Exception as e:
        logger.exception("Errore in /generate_and_predict")
        return JSONResponse(content={"error": str(e)}, status_code=500)
