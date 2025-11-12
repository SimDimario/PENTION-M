from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
import requests
import json
import os

# ============================================================
# API INGESTION SERVICE – Layer 1 (PENTION-M)
# ============================================================

app = FastAPI(title="MLOps Ingestion & Validation Service")

# Percorsi e costanti
LOG_DIR = "/logs"
LOG_FILE = os.path.join(LOG_DIR, "ingestion_log.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# SCHEMA DATI (allineato a sample_simulation_data.json)
# ============================================================

class SensorAir(BaseModel):
    temperature_C: float
    humidity_: float = Field(..., alias="humidity_%")
    wind_speed_mps: float
    wind_dir_deg: int
    stability_class: str

class SensorSubstance(BaseModel):
    compound_name: str
    molecular_formula: Optional[str] = ""
    concentration_series_mg_m3: List[float]
    unit: str
    noise_level: float

class SensorGPS(BaseModel):
    latitude: float
    longitude: float
    altitude_m: float

class PIMLFeatures(BaseModel):
    sigma_y: float
    sigma_z: float
    pe_number: float
    wind_vector: List[float]
    stability_index: float

class Inference(BaseModel):
    dispersion_map_id: str
    predicted_source_location: List[float]
    predicted_class: str
    confidence_score: float

class Monitoring(BaseModel):
    model_version: str
    drift_score: float
    latency_ms: int
    mse_free: float

class ModelOps(BaseModel):
    model_registry_id: str
    training_data_version: str
    retraining_trigger: bool

class UIOutput(BaseModel):
    dashboard_tabs: List[str]
    visualization_files: List[str]

class ForensicExport(BaseModel):
    export_file: str
    hash_sha256: str
    signature: str
    compliance_tags: List[str]

class SimulationData(BaseModel):
    simulation_id: str
    timestamp: str
    SensorAir: SensorAir
    SensorSubstance: SensorSubstance
    SensorGPS: SensorGPS
    PIML_Features: PIMLFeatures
    Inference: Inference
    Monitoring: Monitoring
    ModelOps: ModelOps
    UI_Output: UIOutput
    ForensicExport: ForensicExport

    @validator("timestamp")
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", ""))
        except Exception:
            raise ValueError("timestamp must be ISO 8601")
        return v

# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def append_log(entry: dict):
    """Salva i log in formato JSON lines"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

def safe_post(url: str, payload: dict, label: str):
    """Invia una richiesta POST con gestione sicura degli errori"""
    try:
        r = requests.post(url, json=payload, timeout=60)
        try:
            body = r.json()
        except Exception:
            body = r.text
        print(f"[{label}] -> {r.status_code}: {str(body)[:500]}")
        return body if isinstance(body, dict) else {"status": "error", "raw": body, "code": r.status_code}
    except Exception as e:
        print(f"[WARN] {label} not reachable: {e}")
        return {"status": "warning", "error": str(e)}

# ============================================================
# ENDPOINT PRINCIPALE
# ============================================================

@app.post("/ingest_data")
def ingest_data(sim_data: SimulationData):
    """
    Riceve i dati simulati o reali e li inoltra ai moduli MLOps/PIML.
    """

    data_dict = json.loads(sim_data.json(by_alias=True))

    # Log dell’evento
    append_log({
        "timestamp": datetime.utcnow().isoformat(),
        "simulation_id": sim_data.simulation_id,
        "status": "received",
        "sensor_data": {
            "temperature": sim_data.SensorAir.temperature_C,
            "humidity": sim_data.SensorAir.humidity_,
            "compound": sim_data.SensorSubstance.compound_name
        }
    })

    import numpy as np  # necessario per la generazione dei sensori

    # Inoltro ai moduli PIML (endpoint attuali)
    resp_correction = safe_post(
        "http://correction_dispersion_piml:8008/correct_dispersion",
        {
            "wind_speed": sim_data.SensorAir.wind_speed_mps,
            "wind_dir": [sim_data.SensorAir.wind_dir_deg],
            "concentration_map": [],
            "building_map": [],
            "global_features": [
                sim_data.PIML_Features.sigma_y,
                sim_data.PIML_Features.sigma_z,
                sim_data.PIML_Features.pe_number,
                sim_data.PIML_Features.stability_index
            ]
        },
        label="CorrectionDispersion_PIML"
    )

    # === Costruisci sensori simulati per EmissionSourceLocalization_PIML ===
    payload_sensors = []
    for i in range(5):  # simuliamo 5 sensori attivi
        payload_sensors.append({
            "sensor_id": i + 1,
            "sensor_is_fault": False,
            "time": 0.0,
            "conc": round(sim_data.SensorSubstance.concentration_series_mg_m3[0] * (1 - i * 0.05), 5),
            "wind_dir_x": np.cos(np.radians(sim_data.SensorAir.wind_dir_deg)),
            "wind_dir_y": np.sin(np.radians(sim_data.SensorAir.wind_dir_deg)),
            "wind_speed": sim_data.SensorAir.wind_speed_mps,
            "wind_type": 1,
            "gps_x": sim_data.SensorGPS.longitude + (i * 0.0005),
            "gps_y": sim_data.SensorGPS.latitude + (i * 0.0005),
            "stability_value": sim_data.PIML_Features.stability_index
        })

    resp_localization = safe_post(
        "http://loc_emission_source_piml:8010/predict_source_piml",
        {
            "payload_sensors": payload_sensors,
            "n_sensor_operating": len(payload_sensors)
        },
        label="EmissionSourceLocalization_PIML"
    )

    # === Inoltro ai servizi MLOps (monitoring + forensic) ===

    # Costruzione payload compatibile con MonitoringEvent
    monitoring_payload = {
        "simulation_id": sim_data.simulation_id,
        "timestamp": sim_data.timestamp,
        "SensorAir": {
            "temperature_C": sim_data.SensorAir.temperature_C,
            "humidity_%": sim_data.SensorAir.humidity_,  # alias corretto!
            "wind_speed_mps": sim_data.SensorAir.wind_speed_mps,
            "wind_dir_deg": sim_data.SensorAir.wind_dir_deg,
            "stability_class": sim_data.SensorAir.stability_class
        },
        "PIML_Features": {
            "sigma_y": sim_data.PIML_Features.sigma_y,
            "sigma_z": sim_data.PIML_Features.sigma_z,
            "pe_number": sim_data.PIML_Features.pe_number,
            "wind_vector": sim_data.PIML_Features.wind_vector,
            "stability_index": sim_data.PIML_Features.stability_index
        },
        "Inference": {
            "dispersion_map_id": sim_data.Inference.dispersion_map_id,
            "predicted_source_location": sim_data.Inference.predicted_source_location,
            "predicted_class": sim_data.Inference.predicted_class,
            "confidence_score": sim_data.Inference.confidence_score
        },
        "Monitoring": {
            "model_version": sim_data.Monitoring.model_version,
            "drift_score": sim_data.Monitoring.drift_score,
            "latency_ms": sim_data.Monitoring.latency_ms,
            "mse_free": sim_data.Monitoring.mse_free
        },
        "ModelOps": {
            "model_registry_id": sim_data.ModelOps.model_registry_id,
            "training_data_version": sim_data.ModelOps.training_data_version,
            "retraining_trigger": sim_data.ModelOps.retraining_trigger
        }
    }
    safe_post("http://mlops_monitoring:8012/monitor_event", monitoring_payload, label="Monitoring")

    # Costruzione payload compatibile con ForensicEvent
    forensic_payload = {
        "simulation_id": sim_data.simulation_id,
        "timestamp": sim_data.timestamp,
        "SensorAir": {
            "temperature_C": sim_data.SensorAir.temperature_C,
            "humidity_%": sim_data.SensorAir.humidity_,
            "wind_speed_mps": sim_data.SensorAir.wind_speed_mps,
            "wind_dir_deg": sim_data.SensorAir.wind_dir_deg,
            "stability_class": sim_data.SensorAir.stability_class
        },
        "SensorSubstance": {
            "compound_name": sim_data.SensorSubstance.compound_name,
            "molecular_formula": sim_data.SensorSubstance.molecular_formula,
            "concentration_series_mg_m3": sim_data.SensorSubstance.concentration_series_mg_m3,
            "unit": sim_data.SensorSubstance.unit,
            "noise_level": sim_data.SensorSubstance.noise_level
        },
        "SensorGPS": {
            "latitude": sim_data.SensorGPS.latitude,
            "longitude": sim_data.SensorGPS.longitude,
            "altitude_m": sim_data.SensorGPS.altitude_m
        },
        "PIML_Features": {
            "sigma_y": sim_data.PIML_Features.sigma_y,
            "sigma_z": sim_data.PIML_Features.sigma_z,
            "pe_number": sim_data.PIML_Features.pe_number,
            "stability_index": sim_data.PIML_Features.stability_index
        },
        "Inference": {
            "predicted_class": sim_data.Inference.predicted_class or "",
            "confidence_score": sim_data.Inference.confidence_score or 0.0,
            "dispersion_map_id": sim_data.Inference.dispersion_map_id or ""
        },
        "Monitoring": {
            "model_version": sim_data.Monitoring.model_version or "v1.0",
            "drift_score": sim_data.Monitoring.drift_score or 0.0,
            "latency_ms": sim_data.Monitoring.latency_ms or 0,
            "mse_free": sim_data.Monitoring.mse_free or 0.0
        },
        "ModelOps": {
            "model_registry_id": sim_data.ModelOps.model_registry_id or "",
            "training_data_version": sim_data.ModelOps.training_data_version or "",
            "retraining_trigger": bool(sim_data.ModelOps.retraining_trigger)
        },
        "ForensicExport": {
            "export_file": sim_data.ForensicExport.export_file or "",
            "hash_sha256": sim_data.ForensicExport.hash_sha256 or "",
            "signature": sim_data.ForensicExport.signature or "",
            "compliance_tags": sim_data.ForensicExport.compliance_tags or []
        }
    }
    safe_post("http://mlops_forensic:8013/log_forensic", forensic_payload, label="Forensic")

    return {
        "status": "ok",
        "message": "Simulation data ingested successfully",
        "forwarded_to": {
            "correction_dispersion_piml": resp_correction.get("status", "ok"),
            "source_localization_piml": resp_localization.get("status", "ok"),
        }
    }

# ============================================================
# TEST MANUALE
# ============================================================

if __name__ == "__main__":
    import uvicorn
    from SensorSim_M import SensorSimM

    # Genera un esempio di simulazione e lo invia localmente
    sim = SensorSimM(seed=42)
    payload = sim.generate_simulation()

    print("🧩 Avvio test locale /ingest_data ...")
    uvicorn.run(app, host="0.0.0.0", port=8011)
