from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Any, Dict
from datetime import datetime
import json
import os
import numpy as np
import math
import statistics

# ============================================
# Monitoring & Telemetry Service (Layer 4)
# ============================================

app = FastAPI(title="MLOps Monitoring & Drift Service")

# Paths
LOG_DIR = "/logs"
LOG_FILE = os.path.join(LOG_DIR, "monitoring_log.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------
# Pydantic Schemas
# ---------------------------

class MonitoringBlock(BaseModel):
    model_version: str
    drift_score: float = 0.0
    latency_ms: int = 0
    mse_free: float = 0.0

class InferenceBlock(BaseModel):
    dispersion_map_id: Optional[str] = None
    predicted_source_location: Optional[List[float]] = None
    predicted_class: Optional[str] = None
    confidence_score: Optional[float] = None

class SensorAir(BaseModel):
    temperature_C: Optional[float] = None
    humidity_: Optional[float] = Field(None, alias="humidity_%")
    wind_speed_mps: Optional[float] = None
    wind_dir_deg: Optional[int] = None
    stability_class: Optional[str] = None

class PIMLFeatures(BaseModel):
    sigma_y: Optional[float] = None
    sigma_z: Optional[float] = None
    pe_number: Optional[float] = None
    wind_vector: Optional[List[float]] = None
    stability_index: Optional[float] = None

class MonitoringEvent(BaseModel):
    simulation_id: str
    timestamp: str
    SensorAir: Any = {}
    PIML_Features: Any = {}
    Inference: Any = {}
    Monitoring: Any = {}
    ModelOps: Any = {}

    @validator("timestamp")
    def ts_iso8601(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", ""))
        except Exception:
            raise ValueError("timestamp must be ISO 8601")
        return v

# ---------------------------
# Helpers
# ---------------------------

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def compute_simple_drift(event: MonitoringEvent) -> float:
    """
    Calcola un drift semplificato.
    Supporta sia dict che BaseModel per PIML_Features / Inference.
    """
    score = 0.0
    contrib = 0

    # --- Estrattore generico ---
    def get_attr(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    # --- wind_vector ---
    wv = get_attr(event.PIML_Features, "wind_vector")
    if wv is not None and isinstance(wv, list) and len(wv) >= 1:
        try:
            var = statistics.pvariance(wv)
        except statistics.StatisticsError:
            var = 0.0
        score += min(var / 50.0, 1.0)
        contrib += 1

    # --- stability_index ---
    si = get_attr(event.PIML_Features, "stability_index")
    if si is not None:
        si = safe_float(si, 4.0)
        score += min(abs(si - 3.5) / 3.0, 1.0)
        contrib += 1

    # --- confidence_score ---
    conf = get_attr(event.Inference, "confidence_score")
    if conf is not None:
        conf = safe_float(conf, 1.0)
        score += max(0.0, (0.8 - conf) * 2.0)
        contrib += 1

    if contrib == 0:
        return 0.0
    return max(0.0, min(score / contrib, 1.0))

def compute_latency_trend(events: list[dict]) -> float:
    """Valuta se la latenza media sta crescendo (return 0–1)."""
    if len(events) < 2:
        return 0.0
    last = [e.get("latency_ms", 0) for e in events[-5:]]
    mean_recent = sum(last) / len(last)
    mean_all = sum(e.get("latency_ms", 0) for e in events) / len(events)
    return round(min(max((mean_recent - mean_all) / max(mean_all, 1.0), 0.0), 1.0), 3)

def compute_drift_dynamic(event: MonitoringEvent, history: list[dict]) -> float:
    """
    Drift PIML basato su Mahalanobis multivariato.
    """
    x = build_feature_vector(event)

    baseline = load_baseline()
    mean = baseline["mean"]
    cov = baseline["cov"]
    count = baseline["count"]

    # Se baseline non esiste → inizializza
    MIN_BASELINE_COUNT = 5
    if mean is None or cov is None or count < MIN_BASELINE_COUNT:

        # aggiorna baseline
        if count == 0:
            mean = x
            cov = np.eye(len(x)).tolist()
            count = 1
        else:
            mean = np.array(mean)
            cov = np.array(cov)
            count += 1
            lr = 1.0 / count
            delta = x - mean
            mean = mean + lr * delta
            cov = cov + lr * (np.outer(delta, delta) - cov)

        # convert numpy arrays or lists into serializable lists
        if isinstance(mean, np.ndarray):
            mean_list = mean.tolist()
        else:
            mean_list = list(mean)

        if isinstance(cov, np.ndarray):
            cov_list = cov.tolist()
        else:
            cov_list = list(map(list, cov))

        save_baseline({
            "mean": mean_list,
            "cov": cov_list,
            "count": count,
            "distances": []
        })

        return 0.0  # finché non abbiamo baseline stabile

    # baseline stabile → calcolo Mahalanobis
    mean = np.array(mean)
    cov = np.array(cov)

    # regularizzazione per evitare matrici singolari
    cov = cov + np.eye(len(x)) * 1e-3

    d = mahalanobis(x, mean, cov)

    # aggiorna rolling buffer
    distances = baseline.get("distances", [])
    distances.append(float(d))
    baseline["distances"] = distances

    # calcolo quantile 95%
    if len(distances) < 20:
        # fallback: usiamo la funzione esponenziale
        drift = float(1 - math.exp(-d))
    else:
        q95 = float(np.quantile(distances, 0.95))
        if q95 <= 1e-6:
            drift = float(1 - math.exp(-d))  # fallback safe
        else:
            drift = min(d / q95, 1.0)

    # salva baseline aggiornata
    save_baseline(baseline)

    return round(drift, 4)

def load_last_n(path: str, n: int) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out[-n:] if n > 0 else out

BASELINE_PATH = "/logs/drift_baseline.json"

def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        return {"mean": None, "cov": None, "count": 0, "distances": []}
    try:
        data = json.load(open(BASELINE_PATH))
        # retrocompatibilità con vecchia versione
        if "distances" not in data:
            data["distances"] = []
        return data
    except:
        return {"mean": None, "cov": None, "count": 0, "distances": []}

def save_baseline(bline):
    # limita le distanze agli ultimi 300
    if "distances" in bline and len(bline["distances"]) > 300:
        bline["distances"] = bline["distances"][-300:]
    with open(BASELINE_PATH, "w") as f:
        json.dump(bline, f)

def build_feature_vector(event: MonitoringEvent):
    """
    Feature vector fisico per il drift:
    Supporta sia BaseModel che dict.
    Include:
    - sigma_y, sigma_z, Pe, stability_index
    - confidence_score
    - wind_speed_mps, wind_dir_deg (normalizzato 0–1)
    """
    def get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key, 0.0)
        return getattr(obj, key, 0.0)

    pf = event.PIML_Features or {}
    inf = event.Inference or {}
    sa = event.SensorAir or {}

    sigma_y = safe_float(get(pf, "sigma_y"), 0.0)
    sigma_z = safe_float(get(pf, "sigma_z"), 0.0)
    pe = safe_float(get(pf, "pe_number"), 0.0)
    stab_idx = safe_float(get(pf, "stability_index"), 4.0)
    conf = safe_float(get(inf, "confidence_score"), 1.0)

    wind_speed = safe_float(get(sa, "wind_speed_mps"), 0.0)
    wind_dir = safe_float(get(sa, "wind_dir_deg"), 0.0) / 360.0  # normalizzato

    f = [
        sigma_y,
        sigma_z,
        pe,
        stab_idx,
        conf,
        wind_speed,
        wind_dir,
    ]

    return np.array(f, dtype=np.float32)

def mahalanobis(x, mean, cov):
    try:
        inv = np.linalg.inv(cov)
        d = np.sqrt((x - mean).T @ inv @ (x - mean))
        return float(d)
    except:
        return 0.0


# ---------------------------
# Endpoints
# ---------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "monitoring", "time": datetime.utcnow().isoformat()}

@app.post("/monitor_event")
def monitor_event(event: MonitoringEvent):
    """
    Riceve un evento completo (idealmente lo stesso payload di /ingest_data)
    ed estrae i campi rilevanti per il monitoring. Salva una riga JSONL.
    """
    # Calcola drift se mancante
    history = load_last_n(LOG_FILE, 10)
    drift = compute_drift_dynamic(event, history)
    lat = 0
    mse_free = 0.0
    model_version = "unknown"

    # helper per leggere sia da dict che da oggetto
    def get_attr(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    if event.Monitoring:
        model_version = get_attr(event.Monitoring, "model_version", "unknown")
        lat = get_attr(event.Monitoring, "latency_ms", 0)
        mse_free = get_attr(event.Monitoring, "mse_free", 0.0)
        drift_m = get_attr(event.Monitoring, "drift_score", 0.0)
        drift = max(drift, drift_m)

    stability_index = get_attr(event.PIML_Features, "stability_index", None)
    conf_score = get_attr(event.Inference, "confidence_score", None)

    row = {
        "time": datetime.utcnow().isoformat(),
        "simulation_id": event.simulation_id,
        "model_version": model_version,
        "latency_ms": lat,
        "drift_score": round(drift, 4),
        "mse_free": round(mse_free, 6),
        "stability_index": stability_index,
        "confidence": conf_score,
    }

    append_jsonl(LOG_FILE, row)
    return {"status": "ok", "stored": row}

@app.get("/metrics/summary")
def metrics_summary(last_n: int = Query(200, ge=1, le=5000)):
    """
    Ritorna un riassunto statistico degli ultimi N eventi:
    - media/mediana/min/max di latency e drift
    - conteggio per model_version
    """
    events = load_last_n(LOG_FILE, last_n)
    if not events:
        return {"status": "ok", "count": 0, "summary": {}}

    def collect(field: str):
        vals = [e.get(field) for e in events if isinstance(e.get(field), (int, float))]
        return vals

    lat_vals = collect("latency_ms")
    drift_vals = collect("drift_score")
    mse_vals = collect("mse_free")

    def stats(vals: List[float]) -> dict:
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "mean": round(float(sum(vals) / len(vals)), 4),
            "median": round(float(statistics.median(vals)), 4),
            "min": round(float(min(vals)), 4),
            "max": round(float(max(vals)), 4),
        }

    versions: Dict[str, int] = {}
    for e in events:
        v = e.get("model_version", "unknown")
        versions[v] = versions.get(v, 0) + 1

    return {
        "status": "ok",
        "count": len(events),
        "summary": {
            "latency_ms": stats(lat_vals),
            "drift_score": stats(drift_vals),
            "mse_free": stats(mse_vals),
            "by_model_version": versions
        }
    }

@app.get("/metrics/last")
def last_events(k: int = Query(10, ge=1, le=200)):
    """Restituisce le ultime k righe grezze del monitoring log."""
    return {"status": "ok", "items": load_last_n(LOG_FILE, k)}

# ---------------------------
# Local run
# ---------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)
