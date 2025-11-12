from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Any, Dict
from datetime import datetime
import json
import os
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
    drift_calculated = compute_simple_drift(event)
    drift = drift_calculated
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
