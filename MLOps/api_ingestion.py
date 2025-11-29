from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from typing import Optional
import requests
import json
import os
import sys
import time
import hashlib

MODEL_REGISTRY_PATH = "/logs/model_registry.json"

import numpy as np

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
    concentration_series_mg_m3: List[float]
    unit: str
    noise_level: float

class SensorGPS(BaseModel):
    latitude: float
    longitude: float
    altitude_m: float

class SourceGPS(BaseModel):
    latitude: float
    longitude: float

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
    compliance_tags: List[str]

class SimulationData(BaseModel):
    simulation_id: str
    timestamp: str
    event_start_ts: Optional[str] = None
    SensorAir: SensorAir
    SensorSubstance: SensorSubstance
    SensorGPS: SensorGPS
    SourceGPS: SourceGPS

    # valori calcolati da ingestion → NON obbligatori in ingresso
    PIML_Features: Optional[PIMLFeatures] = None
    Inference: Optional[Inference] = None
    Monitoring: Optional[Monitoring] = None
    ModelOps: Optional[ModelOps] = None
    UI_Output: Optional[UIOutput] = None
    ForensicExport: Optional[ForensicExport] = None

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

def generate_concentration_map_from_gaussian(sensor_air: dict, src_x: int, src_y: int):
    """
    Esegue una simulazione GaussianPuff e restituisce:
    - conc_map_2d: mappa media nel tempo (H x W, normalizzata 0–1)
    - conc_time_series: media spaziale nel tempo (len = n_step temporali)
    - C1: tensor completo (x, y, t)
    - stability_array: serie temporale della stabilità (valori Pasquill-Gifford)
    - wind_dir_series: serie temporale della direzione del vento (deg)
    Usa realmente:
      - wind_speed_mps
      - wind_dir_deg
      - stability_class (A–F → PasquillGiffordStability)
      - humidity (RH)
    """
    try:
        # mappa classe di stabilità (A–F) → Pasquill-Gifford
        stab_map = {
            "A": PasquillGiffordStability.VERY_UNSTABLE,
            "B": PasquillGiffordStability.MODERATELY_UNSTABLE,
            "C": PasquillGiffordStability.SLIGHTLY_UNSTABLE,
            "D": PasquillGiffordStability.NEUTRAL,
            "E": PasquillGiffordStability.MODERATELY_STABLE,
            "F": PasquillGiffordStability.VERY_STABLE,
        }
        stab_key = str(sensor_air.get("stability_class", "D")).upper()
        stab_value = stab_map.get(stab_key, PasquillGiffordStability.NEUTRAL)

        wind_speed = float(sensor_air.get("wind_speed_mps", 4.0))
        wind_dir_deg = float(sensor_air.get("wind_dir_deg", 225.0))

        # Umidità relativa se disponibile
        rh = float(sensor_air.get("humidity", 0.6))

        config = ModelConfig(
            days=1,
            RH=rh,
            aerosol_type=None,  # oppure tipo reale se vorrai
            humidify=False,
            stability_profile=StabilityType.CONSTANT,
            stability_value=stab_value,
            wind_type=WindType.CONSTANT,
            wind_speed=wind_speed,
            output=OutputType.PLAN_VIEW,
            dispersion_model=DispersionModelType.PLUME,
            stacks=[(src_x, src_y, 10.0, 20.0)],
            grid_size=500,
            wind_dir_deg=wind_dir_deg,
        )

        C1, (x_grid, y_grid, z_grid), times, stability_array, wind_dir_series, *_ = run_dispersion_model(config)

        # mappa media nel tempo (H x W)
        conc_map_2d = np.mean(C1, axis=2)
        p95 = np.percentile(conc_map_2d, 95)
        conc_map_2d = conc_map_2d / (p95 + 1e-6)
        conc_map_2d = np.clip(conc_map_2d, 0, 1)

        # serie temporale (media su spazio, funzione del tempo)
        conc_time_series = np.mean(C1, axis=(0, 1))

        return conc_map_2d.tolist(), conc_time_series.tolist(), C1, stability_array, wind_dir_series

    except Exception as e:
        print(f"[WARN] GaussianPuff simulation failed: {e}")
        # fallback safe: nessuna dinamica → stabilità neutra, vento fisso
        return (
            np.zeros((500, 500)).tolist(),
            [0.0],
            np.zeros((500, 500, 1)),
            np.array([4.0], dtype=np.float32),         # stability_index ~ neutro
            np.array([225.0], dtype=np.float32),       # direzione SW
        )

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

def sanitize_correction_response(resp):
    """
    Rimuove la corrected_map dal runtime da salvare nel bundle forense
    e la sostituisce con metadati leggeri (shape + hash).
    """
    if not isinstance(resp, dict):
        return resp

    clean = {k: v for k, v in resp.items()
             if k not in ("corrected_map", "corrected_concentration_map")}

    cm = resp.get("corrected_map") or resp.get("corrected_concentration_map")
    if cm is not None:
        try:
            arr = np.array(cm, dtype=np.float32)
            clean["corrected_map_shape"] = list(arr.shape)
            clean["corrected_map_hash"] = hashlib.sha256(arr.tobytes()).hexdigest()
        except Exception as e:
            clean["corrected_map_error"] = f"hash_failed: {e}"

    return clean

# ============================================================
# ENDPOINT PRINCIPALE
# ============================================================

@app.post("/ingest_data")
def ingest_data(sim_data: SimulationData):
    """
    Riceve i dati simulati o reali e li inoltra ai moduli PIML/MLOps.
    Qui avviene la vera catena fisica + ML:

    UI → (SensorAir, SensorGPS, dati di base)
       → GaussianPuff (mappa di concentrazione)
       → CorrectionDispersion_PIML (mappa corretta + versioning)
       → SensorSim_M (rete sensori)
       → EmissionSourceLocalization_PIML (stima sorgente)
       → Monitoring (drift + latency reali)
       → Forensic (bundle firmato con artifact hash)
    """
    full_start_ts = sim_data.event_start_ts

    # ============= 0. LOG INIZIALE =============
    append_log({
        "timestamp": datetime.utcnow().isoformat(),
        "simulation_id": sim_data.simulation_id,
        "status": "received",
        "model_version_used": (sim_data.Monitoring.model_version if sim_data.Monitoring else "unknown"),
        "sensor_data": {
            "temperature": sim_data.SensorAir.temperature_C,
            "humidity": sim_data.SensorAir.humidity_,
            "compound": sim_data.SensorSubstance.compound_name
        }
    })

    # ============= 1. GAUSSIAN PUFF: MAPPA DI CONCENTRAZIONE =============
    # Bounding box WGS84 della UI (Amsterdam)
    LAT_MIN = 52.35
    LAT_MAX = 52.39
    LON_MIN = 4.88
    LON_MAX = 4.92
    GRID = 500

    def latlon_to_grid(lat, lon):
        gx = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * (GRID - 1))
        gy = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * (GRID - 1))
        gx = max(0, min(GRID - 1, gx))
        gy = max(0, min(GRID - 1, gy))
        return gx, gy

    def grid_to_latlon(gx, gy):
        lat = LAT_MIN + gy / (GRID - 1) * (LAT_MAX - LAT_MIN)
        lon = LON_MIN + gx / (GRID - 1) * (LON_MAX - LON_MIN)
        return lat, lon


    # Convertiamo la posizione reale (lat/lon) in coordinate locali (0..499)
    src_x, src_y = latlon_to_grid(sim_data.SourceGPS.latitude, sim_data.SourceGPS.longitude)

    # Eseguiamo GaussianPuff usando la vera sorgente convertita
    conc_map_real, conc_time_series, C1, stability_array, wind_dir_series = generate_concentration_map_from_gaussian(
            {
                "wind_speed_mps": sim_data.SensorAir.wind_speed_mps,
                "wind_dir_deg": sim_data.SensorAir.wind_dir_deg,
                "stability_class": sim_data.SensorAir.stability_class,
                "humidity": sim_data.SensorAir.humidity_,
            },
            src_x,
            src_y
    )

    from gaussianPuff.sigmaCalculation import calc_sigmas

    distances = np.sqrt((np.arange(500)[:,None] - src_y)**2 + 
                        (np.arange(500)[None,:] - src_x)**2) * 10  # scala metri

    stab_class = sim_data.SensorAir.stability_class.upper()
    pg_map = { "A":1, "B":2, "C":3, "D":4, "E":5, "F":6 }
    pg_category = pg_map.get(stab_class, 4)

    sigma_y_map, sigma_z_map = calc_sigmas(pg_category, distances)

    sigma_y = float(np.mean(sigma_y_map))
    sigma_z = float(np.mean(sigma_z_map))

    # Péclet number semplificato
    pe_number = sim_data.SensorAir.wind_speed_mps / (sigma_y + 1e-6)

    # stability_index fisico: media della serie di stabilità di GaussianPuff
    try:
        if isinstance(stability_array, np.ndarray):
            stability_index = float(np.mean(stability_array))
        else:
            stability_index = float(np.mean(stability_array))
    except Exception:
        # fallback sulla classe A–F se qualcosa va storto
        stability_index = stability_to_index(sim_data.SensorAir.stability_class)

    # Direzione vento media dal modello (se disponibile), altrimenti da UI
    try:
        if isinstance(wind_dir_series, np.ndarray):
            wind_dir_deg_eff = float(np.mean(wind_dir_series))
        else:
            wind_dir_deg_eff = float(sim_data.SensorAir.wind_dir_deg)
    except Exception:
        wind_dir_deg_eff = float(sim_data.SensorAir.wind_dir_deg)

    # Se la UI non ha mandato PIML_Features → li generiamo noi (physics-informed)
    sim_data.PIML_Features = PIMLFeatures(
        sigma_y=sigma_y,
        sigma_z=sigma_z,
        pe_number=pe_number,
        wind_vector=[
            float(np.cos(np.radians(wind_dir_deg_eff))),
            float(np.sin(np.radians(wind_dir_deg_eff))),
        ],
        stability_index=stability_index
    )

    # ============= 2. CORRECTION DISPERSION PIML =============
    # Carichiamo la binary map reale (edifici Amsterdam) se disponibile
    try:
        from CorrectionDispersion_PIML.api_correction_piml import DEFAULT_BUILDING_MAP
        building_map_real = DEFAULT_BUILDING_MAP.tolist()
    except Exception:
        building_map_real = []

    # payload base per PIML
    correction_payload = {
        "wind_speed": sim_data.SensorAir.wind_speed_mps,
        "wind_dir": [sim_data.SensorAir.wind_dir_deg],
        "concentration_map": conc_map_real,
        "building_map": building_map_real,
    }

    # === TODO3: passiamo anche il TENSORE 3D completo C1 (x, y, t) ===
    # Questo rende il modulo PIML davvero "physics-informed", perché vede la dinamica temporale.
    try:
        correction_payload["concentration_tensor_3d"] = C1.tolist()
    except Exception as e:
        print(f"[WARN] Impossibile serializzare C1 in JSON: {e}")

    resp_correction = safe_post(
        "http://correction_dispersion_piml:8008/correct_dispersion",
        correction_payload,
        label="CorrectionDispersion_PIML",
    )

    # Se il servizio PIML restituisce una mappa corretta, usiamo quella;
    # altrimenti ricadiamo sulla mappa GaussianPuff.
    corrected_map = None
    if isinstance(resp_correction, dict):
        corrected_map = resp_correction.get("corrected_map") or resp_correction.get("corrected_concentration_map")

    if corrected_map is None:
        corrected_map = conc_map_real

    conc_map_np = np.array(corrected_map, dtype=np.float32)
    building_map_np = (
        np.array(building_map_real, dtype=np.float32)
        if len(building_map_real) > 0
        else np.zeros_like(conc_map_np)
    )

    # === RMSE tra mappa GaussianPuff e mappa corretta PIML ===
    diff_sq = (np.array(conc_map_real, dtype=np.float32) - conc_map_np) ** 2
    mse_free = float(np.sqrt(np.mean(diff_sq)))

    # ============= 3. SENSOR NETWORK + SOURCE LOCALIZATION PIML =============
    # Generiamo sensori virtuali campionando dalla mappa 2D corretta.
    payload_sensors = []

    van_series = C1[src_y, src_x, :].tolist()
    van_time = list(range(len(van_series)))

    payload_sensors.append({
        "sensor_id": 1,
        "sensor_is_fault": False,        # ⬅ OBBLIGATORIA
        "time": van_time,
        "conc": van_series,
        "concentration_series": van_series,
        "wind_dir_x": np.cos(np.radians(wind_dir_deg_eff)),
        "wind_dir_y": np.sin(np.radians(wind_dir_deg_eff)),
        "wind_speed": sim_data.SensorAir.wind_speed_mps,
        "wind_type": 1,
        "gps_x": src_x,
        "gps_y": src_y,
        "stability_value": stability_index,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
        "pe_number": pe_number,
    })


    resp_localization = safe_post(
        "http://loc_emission_source_piml:8010/predict_source_piml",
        {
            "payload_sensors": payload_sensors,
            "n_sensor_operating": 1
        },
        label="EmissionSourceLocalization_PIML",
    )

    # Estraggo, se presente, una stima della sorgente da resp_localization
    def extract_source_xy(resp: dict):
        if not isinstance(resp, dict):
            return None
        for key in ["predicted_source_xy", "predicted_source", "source_xy", "source"]:
            coords = resp.get(key)
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                try:
                    return [float(coords[0]), float(coords[1])]
                except Exception:
                    pass
        return None

    predicted_source_xy = extract_source_xy(resp_localization)

    pred_lat = None
    pred_lon = None

    if predicted_source_xy is not None:
        try:
            px, py = predicted_source_xy
            pred_lat, pred_lon = grid_to_latlon(px, py)
        except Exception:
            pred_lat, pred_lon = None, None

    # 1) Preinizializzazione
    predicted_class = sim_data.SensorSubstance.compound_name
    confidence_clf = 0.0

    # 2) Se Inference non esiste → crealo
    dispersion_map_id = hashlib.sha256(
        np.array(conc_map_real, dtype=np.float32).tobytes()
    ).hexdigest()

    if sim_data.Inference is None:
        sim_data.Inference = Inference(
            dispersion_map_id=dispersion_map_id,
            predicted_source_location=(
                predicted_source_xy if predicted_source_xy is not None else [0.0, 0.0]
            ),
            predicted_class=predicted_class,
            confidence_score=confidence_clf
        )

    # 3) Ora puoi aggiornare la sorgente stimata
    if predicted_source_xy is not None:
        sim_data.Inference.predicted_source_location = predicted_source_xy

    # ============= 4. CLASSIFICATORE NPS — USA LO SPETTRO DELLA UI =============
    spectrum_ui_original = np.array(sim_data.SensorSubstance.concentration_series_mg_m3)
    try:
        spectrum_noisy = spectrum_ui_original.copy()

        # Se lo spettro non è lungo 600 elementi → fallback soft
        if spectrum_noisy.shape[0] != 600:
            print("[WARN] UI spectrum size != 600. Padding/cropping applied.")
            if spectrum_noisy.shape[0] > 600:
                spectrum_noisy = spectrum_noisy[:600]
            else:
                spectrum_noisy = np.pad(spectrum_noisy, (0, 600 - len(spectrum_noisy)))

        # INVIO AL VERO CLASSIFICATORE
        resp_nps = safe_post(
            "http://clas_nps:8000/predict_xgb",
            {"spectra": [spectrum_noisy.tolist()]},
            label="ClassificatoreNPS (XGB)"
        )

        if isinstance(resp_nps, dict):
            predicted_class = resp_nps.get("predictions", [sim_data.SensorSubstance.compound_name])[0]
            confidence_clf = float(resp_nps.get("confidence", 0.0))

        sim_data.Inference.predicted_class = predicted_class
        sim_data.Inference.confidence_score = confidence_clf

    except Exception as e:
        print(f"[WARN] NPS classification failed: {e}")


    # ============= 5. MODEL REGISTRY E VERSIONING =============
    effective_model_version = "PIML_v1"  # default se registry assente

    if os.path.exists(MODEL_REGISTRY_PATH):
        try:
            with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
                reg = json.load(f)

            # forza reload del modello ogni richiesta (PENTION-M è realtime)
            import importlib
            if "service_correction_piml" in sys.modules:
                importlib.reload(sys.modules["service_correction_piml"])

            v = reg.get("current_model_version")
            if v:
                effective_model_version = v
                print(f"[INFO] Ingestion usa model_version reale: {effective_model_version}")
        except Exception as e:
            print(f"[WARN] Errore lettura registry: {e}")
    else:
        print("Registry assente → default PIML_v1")
        reg = {"training_data_version": "PIML_DS_v1", "metrics": {}}

    # Se il blocco Monitoring non esiste (UI non lo manda), inizializzalo
    if sim_data.Monitoring is None:
        sim_data.Monitoring = Monitoring(
            model_version=effective_model_version,
            drift_score=0.0,
            latency_ms=0,
            mse_free=0.0
        )
    else:
        sim_data.Monitoring.model_version = effective_model_version

    # Se ModelOps non è presente → creiamo defaults
    if sim_data.ModelOps is None:
        td_version = reg.get("training_data_version", "PIML_DS_v1")
        sim_data.ModelOps = ModelOps(
            model_registry_id="mdl_pention_m",
            training_data_version=td_version,
            retraining_trigger=False
        )
    else:
        # aggiorna sempre in base al registry
        td_version = reg.get("training_data_version", "PIML_DS_v1")
        sim_data.ModelOps.training_data_version = td_version

    # 6-BIS) CALCOLO LATENCY END-TO-END (UI → INGESTION → TUTTA PIPELINE)
    if full_start_ts:
        try:
            t0 = datetime.fromisoformat(full_start_ts.replace("Z", ""))
            t1 = datetime.utcnow()
            latency_full_ms = int((t1 - t0).total_seconds() * 1000)
        except Exception:
            latency_full_ms = 0
    else:
        latency_full_ms = 0

    # ============= 6. MONITORING: DRIFT + LATENZA REALE =============
    monitoring_payload = {
        "simulation_id": sim_data.simulation_id,
        "timestamp": sim_data.timestamp,
        "SensorAir": {
            "temperature_C": sim_data.SensorAir.temperature_C,
            "humidity_%": sim_data.SensorAir.humidity_,
            "wind_speed_mps": sim_data.SensorAir.wind_speed_mps,
            "wind_dir_deg": sim_data.SensorAir.wind_dir_deg,
            "stability_class": sim_data.SensorAir.stability_class,
        },
        "PIML_Features": {
            "sigma_y": sim_data.PIML_Features.sigma_y,
            "sigma_z": sim_data.PIML_Features.sigma_z,
            "pe_number": sim_data.PIML_Features.pe_number,
            "wind_vector": sim_data.PIML_Features.wind_vector,
            "stability_index": sim_data.PIML_Features.stability_index,
        },
        "Inference": {
            "dispersion_map_id": sim_data.Inference.dispersion_map_id,
            "predicted_source_location": (
                predicted_source_xy
                if predicted_source_xy is not None
                else sim_data.Inference.predicted_source_location
            ),
            "predicted_class": predicted_class,
            "confidence_score": confidence_clf,
        },
        "Monitoring": {
            "model_version": sim_data.Monitoring.model_version,
            "drift_score": (sim_data.Monitoring.drift_score if sim_data.Monitoring else 0.0),
            "latency_ms": latency_full_ms,
            "mse_free": mse_free,
        },
        "ModelOps": {
            "model_registry_id": sim_data.ModelOps.model_registry_id,
            "training_data_version": sim_data.ModelOps.training_data_version,
            "retraining_trigger": sim_data.ModelOps.retraining_trigger,
        },
    }

    # 7) CHIAMATA AL MONITORING SERVICE CON LATENCY END-TO-END
    resp_monitoring = safe_post(
        "http://mlops_monitoring:8012/monitor_event",
        monitoring_payload,
        label="Monitoring",
    )

    # Recuperiamo il drift calcolato dal monitoring service
    if isinstance(resp_monitoring, dict):
        drift_value = (
            resp_monitoring.get("stored", {}).get("drift_score")
            or sim_data.Monitoring.drift_score
        )
    else:
        drift_value = sim_data.Monitoring.drift_score

    print(f"[INFO] Latenza end-to-end: {latency_full_ms} ms, drift: {drift_value}")

    # Aggiorno il payload interno
    monitoring_payload["Monitoring"]["latency_ms"] = latency_full_ms
    monitoring_payload["Monitoring"]["drift_score"] = drift_value

    # Aggiorno il blocco Monitoring interno
    sim_data.Monitoring.latency_ms = latency_full_ms
    sim_data.Monitoring.drift_score = drift_value
    sim_data.Monitoring.mse_free = mse_free

    # === MODEL OPS: trigger di retraining basato su drift ===
    # soglia arbitraria ma chiara per la tesi (es. 0.3)
    retrain = drift_value is not None and drift_value > 0.3
    sim_data.ModelOps.retraining_trigger = bool(retrain)

    monitoring_out = {
        "simulation_id": sim_data.simulation_id,
        "model_version": sim_data.Monitoring.model_version,
        "latency_ms": latency_full_ms,
        "drift_score": drift_value,
        "stability_index": sim_data.PIML_Features.stability_index,
        "confidence": confidence_clf,
    }

    # ============= 7. FORENSIC: COSTRUZIONE EVENTO COMPLETO =============

    # Se ForensicExport non è presente → creiamo export con compliance "reale"
    if sim_data.ForensicExport is None:
        compliance_tags = []

        if drift_value is not None and drift_value > 0.3:
            compliance_tags.append("DRIFT_HIGH")
        else:
            compliance_tags.append("DRIFT_OK")

        if latency_full_ms > 500:
            compliance_tags.append("LATENCY_HIGH")
        else:
            compliance_tags.append("LATENCY_OK")

        compliance_tags.append("GDPR_SIMULATION_ONLY")

        sim_data.ForensicExport = ForensicExport(
            export_file=f"{sim_data.simulation_id}_bundle.zip",
            compliance_tags=compliance_tags
        )

    forensic_payload = {
        "simulation_id": sim_data.simulation_id,
        "timestamp": sim_data.timestamp,
        "SensorAir": {
            "temperature_C": sim_data.SensorAir.temperature_C,
            "humidity_%": sim_data.SensorAir.humidity_,
            "wind_speed_mps": sim_data.SensorAir.wind_speed_mps,
            "wind_dir_deg": sim_data.SensorAir.wind_dir_deg,
            "stability_class": sim_data.SensorAir.stability_class,
        },
        "SensorSubstance": {
            "compound_name": sim_data.SensorSubstance.compound_name,
            "spectrum_ei_1_600": spectrum_ui_original.tolist(),
            "concentration_series_mg_m3": sim_data.SensorSubstance.concentration_series_mg_m3,
            "unit": "EI_intensity",
            "noise_level": sim_data.SensorSubstance.noise_level,
        },
        "SensorGPS": {
            "latitude": sim_data.SensorGPS.latitude,
            "longitude": sim_data.SensorGPS.longitude,
            "altitude_m": sim_data.SensorGPS.altitude_m,
        },
        "SourceGPS": {
            "latitude": sim_data.SourceGPS.latitude,
            "longitude": sim_data.SourceGPS.longitude
        },
        "PIML_Features": {
            "sigma_y": sim_data.PIML_Features.sigma_y,
            "sigma_z": sim_data.PIML_Features.sigma_z,
            "pe_number": sim_data.PIML_Features.pe_number,
            "stability_index": sim_data.PIML_Features.stability_index,
            "wind_vector": sim_data.PIML_Features.wind_vector,
        },
        "Inference": {
            "predicted_class": predicted_class or "",
            "confidence_score": confidence_clf or 0.0,
            "dispersion_map_id": sim_data.Inference.dispersion_map_id or "",
            "predicted_source_location": (
                predicted_source_xy
                if predicted_source_xy is not None
                else sim_data.Inference.predicted_source_location
            ),
        },
        "inference_latlon": {
            "latitude": pred_lat if pred_lat is not None else None,
            "longitude": pred_lon if pred_lon is not None else None
        },
        "Monitoring": {
            "model_version": sim_data.Monitoring.model_version or "v1.0",
            "drift_score": drift_value,
            "latency_ms": latency_full_ms,
            "mse_free": mse_free,
        },
        "ModelOps": {
            "model_registry_id": sim_data.ModelOps.model_registry_id or "",
            "training_data_version": sim_data.ModelOps.training_data_version or "",
            "retraining_trigger": bool(sim_data.ModelOps.retraining_trigger),
        },
        "ForensicExport": {
            "export_file": sim_data.ForensicExport.export_file or "",
            "compliance_tags": sim_data.ForensicExport.compliance_tags or [],
        },
        "SensorNetworkTimeSeries": payload_sensors,
        # metadati grezzi dei servizi PIML → utili per audit/analisi offline
        "PIML_Runtime": {
            "correction_dispersion_piml": sanitize_correction_response(resp_correction),
            "source_localization_piml": resp_localization,
        },
    }

    safe_post(
        "http://mlops_forensic:8013/log_forensic",
        forensic_payload,
        label="Forensic",
    )

    # Risposta verso la UI
    return {
        "status": "ok",
        "message": "Simulation data ingested successfully",
        "monitoring": monitoring_out,
        "forwarded_to": {
            "correction_dispersion_piml": resp_correction.get("status", "ok")
            if isinstance(resp_correction, dict)
            else "unknown",
            "source_localization_piml": resp_localization.get("status", "ok")
            if isinstance(resp_localization, dict)
            else "unknown",
        },
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
