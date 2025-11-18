from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
import requests
import json
import os
import sys
import time
MODEL_REGISTRY_PATH = "/logs/model_registry.json"

import numpy as np
from SensorSim_M import generate_sensor_network_from_map

# --- Fix import path per gaussianPuff (garantito) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append("/gaussianPuff")
sys.path.append("/MLOps")

# --- Collegamento GaussianPuff ---
from gaussianPuff.gaussianModel import run_dispersion_model
from gaussianPuff.config import ModelConfig, StabilityType, WindType, OutputType, DispersionModelType, PasquillGiffordStability

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

def stability_to_index(stab: str) -> float:
    mapping = {
        "A": 1.0,
        "B": 2.0,
        "C": 3.0,
        "D": 4.0,
        "E": 5.0,
        "F": 6.0
    }
    return mapping.get(stab.upper(), 4.0)  # default = neutral

def generate_concentration_map_from_gaussian(sensor_air: dict):
    """
    Esegue una simulazione GaussianPuff e restituisce la mappa di concentrazione 2D.
    """
    try:
        # Configurazione base del modello
        config = ModelConfig(
            days=1,
            RH=0.6,  # umidità relativa media
            aerosol_type=None,  # o NPS.CATHINONE_ANALOGUES
            humidify=False,
            stability_profile=StabilityType.CONSTANT,
            stability_value=PasquillGiffordStability.SLIGHTLY_UNSTABLE,
            wind_type=WindType.CONSTANT,
            wind_speed=sensor_air["wind_speed_mps"],
            output=OutputType.PLAN_VIEW,
            dispersion_model=DispersionModelType.PLUME,
            stacks=[(0, 0, 10.0, 20.0)],
            grid_size=500
        )

        C1, (x_grid, y_grid, z_grid), *_ = run_dispersion_model(config)
        # Prendiamo la media temporale come mappa 2D (ground-level)
        conc_map = np.mean(C1, axis=2)
        # Normalizziamo tra 0–1 per stabilità numerica
        conc_map = conc_map / np.max(conc_map + 1e-8)
        return conc_map.tolist()

    except Exception as e:
        print(f"[WARN] GaussianPuff simulation failed: {e}")
        return np.zeros((500, 500)).tolist()

def append_log(entry: dict):
    """Salva i log in formato JSON lines"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

def safe_post(url: str, payload: dict, label: str):
    """Invia una richiesta POST con gestione sicura degli errori"""
    try:
        r = requests.post(url, json=payload, timeout=180)
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
        "model_version_used": sim_data.Monitoring.model_version,
        "sensor_data": {
            "temperature": sim_data.SensorAir.temperature_C,
            "humidity": sim_data.SensorAir.humidity_,
            "compound": sim_data.SensorSubstance.compound_name
        }
    })

    import numpy as np  # necessario per la generazione dei sensori

    # Inoltro ai moduli PIML (endpoint attuali)
    # === Esecuzione GaussianPuff e generazione mappa fisica ===
    conc_map_real = generate_concentration_map_from_gaussian({
        "wind_speed_mps": sim_data.SensorAir.wind_speed_mps
    })

    # Carica mappa edifici reale se disponibile
    try:
        from CorrectionDispersion_PIML.api_correction_piml import DEFAULT_BUILDING_MAP
        building_map_real = DEFAULT_BUILDING_MAP.tolist()
    except Exception:
        building_map_real = []

    # === Invio reale al servizio CorrectionDispersion_PIML ===
    resp_correction = safe_post(
        "http://correction_dispersion_piml:8008/correct_dispersion",
        {
            "wind_speed": sim_data.SensorAir.wind_speed_mps,
            "wind_dir": [sim_data.SensorAir.wind_dir_deg],
            "concentration_map": conc_map_real,
            "building_map": building_map_real,
            "global_features": [
                sim_data.PIML_Features.sigma_y,
                sim_data.PIML_Features.sigma_z,
                sim_data.PIML_Features.pe_number,
                sim_data.PIML_Features.stability_index
            ]
        },
        label="CorrectionDispersion_PIML"
    )

    # === Generazione sensori fisici campionati dalla mappa ===
    conc_map_np = np.array(conc_map_real)
    building_map_np = np.array(building_map_real) if len(building_map_real) > 0 else np.zeros_like(conc_map_np)
    payload_sensors = generate_sensor_network_from_map(conc_map_np, building_map_np, n_sensors=5, fault_rate=0.1, seed=42)

    resp_localization = safe_post(
        "http://loc_emission_source_piml:8010/predict_source_piml",
        {
            "payload_sensors": payload_sensors,
            "n_sensor_operating": len(payload_sensors)
        },
        label="EmissionSourceLocalization_PIML"
    )

    # === Inoltro ai servizi MLOps (monitoring + forensic) ===
    # === Carica la versione modello dal registry e forza la versione utilizzata ===
    effective_model_version = "PIML_v1"  # default iniziale se registry assente

    if os.path.exists(MODEL_REGISTRY_PATH):
        try:
            with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
                reg = json.load(f)
            v = reg.get("current_model_version")
            if v:
                effective_model_version = v
                print(f"[INFO] 🔄 Ingestion usa model_version reale: {effective_model_version}")
        except Exception as e:
            print(f"[WARN] Errore lettura registry: {e}")
    else:
        print("Registry assente → default PIML_v1")

    # Forziamo il model_version usato per monitoring
    sim_data.Monitoring.model_version = effective_model_version

    # --- Calcolo latenza reale e invio ai servizi MLOps ---

    # 1) Costruisco il payload SENZA drift_value e latency (placeholder)
    monitoring_payload = {
        "simulation_id": sim_data.simulation_id,
        "timestamp": sim_data.timestamp,
        "SensorAir": {
            "temperature_C": sim_data.SensorAir.temperature_C,
            "humidity_%": sim_data.SensorAir.humidity_,
            "wind_speed_mps": sim_data.SensorAir.wind_speed_mps,
            "wind_dir_deg": sim_data.SensorAir.wind_dir_deg,
            "stability_class": sim_data.SensorAir.stability_class
        },
        "PIML_Features": {
            "sigma_y": sim_data.PIML_Features.sigma_y,
            "sigma_z": sim_data.PIML_Features.sigma_z,
            "pe_number": sim_data.PIML_Features.pe_number,
            "wind_vector": sim_data.PIML_Features.wind_vector,
            "stability_index": stability_to_index(sim_data.SensorAir.stability_class)
        },
        "Inference": {
            "dispersion_map_id": sim_data.Inference.dispersion_map_id,
            "predicted_source_location": sim_data.Inference.predicted_source_location,
            "predicted_class": sim_data.Inference.predicted_class,
            "confidence_score": sim_data.Inference.confidence_score
        },
        "Monitoring": {
            "model_version": sim_data.Monitoring.model_version,
            "drift_score": sim_data.Monitoring.drift_score,     # TEMP
            "latency_ms": sim_data.Monitoring.latency_ms,       # TEMP
            "mse_free": sim_data.Monitoring.mse_free
        },
        "ModelOps": {
            "model_registry_id": sim_data.ModelOps.model_registry_id,
            "training_data_version": sim_data.ModelOps.training_data_version,
            "retraining_trigger": sim_data.ModelOps.retraining_trigger
        }
    }

    # 2) Misuro la latenza della POST /monitor_event
    t0 = time.time()
    resp_monitoring = safe_post(
        "http://mlops_monitoring:8012/monitor_event",
        monitoring_payload,
        label="Monitoring"
    )
    t1 = time.time()

    latency_ms = round((t1 - t0) * 1000, 2)

    # 3) Recupero drift calcolato dal monitoring service
    if isinstance(resp_monitoring, dict):
        drift_value = resp_monitoring.get("stored", {}).get("drift_score")
    else:
        drift_value = None

    # fallback se nulla arrivasse
    if drift_value is None:
        drift_value = sim_data.Monitoring.drift_score

    print(f"[INFO] Latenza reale misurata: {latency_ms} ms, drift: {drift_value}")

    # 4) Aggiorno il payload interno (per forensic e UI)
    monitoring_payload["Monitoring"]["latency_ms"] = latency_ms
    monitoring_payload["Monitoring"]["drift_score"] = drift_value

    # 5) Oggetto monitoring per la UI
    monitoring_out = {
        "simulation_id": sim_data.simulation_id,
        "model_version": sim_data.Monitoring.model_version,
        "latency_ms": latency_ms,
        "drift_score": drift_value,
        "stability_index": stability_to_index(sim_data.SensorAir.stability_class),
        "confidence": sim_data.Inference.confidence_score,
    }

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
            "stability_index": stability_to_index(sim_data.SensorAir.stability_class)
        },
        "Inference": {
            "predicted_class": sim_data.Inference.predicted_class or "",
            "confidence_score": sim_data.Inference.confidence_score or 0.0,
            "dispersion_map_id": sim_data.Inference.dispersion_map_id or ""
        },
        "Monitoring": {
            "model_version": sim_data.Monitoring.model_version or "v1.0",
            "drift_score": drift_value,
            "latency_ms": latency_ms,
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
        "monitoring": monitoring_out,   # <-- aggiunto
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

    sim = SensorSimM(seed=42)
    payload = sim.generate_simulation()

    print("Avvio test locale /ingest_data ...")
    uvicorn.run(app, host="0.0.0.0", port=8011)
