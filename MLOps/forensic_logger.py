from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import json
import os
import uuid

# ============================================================
# Forensic Logger & ModelOps Service (Layer 5)
# ============================================================

app = FastAPI(title="MLOps Forensic Logger & ModelOps")

# Percorsi e setup directory
LOG_DIR = "/logs"
FORENSIC_DIR = os.path.join(LOG_DIR, "forensic")
os.makedirs(FORENSIC_DIR, exist_ok=True)

# ============================================================
# MODELLI DATI
# ============================================================

class ForensicExport(BaseModel):
    export_file: str
    hash_sha256: str
    signature: str
    compliance_tags: List[str]

class ModelOps(BaseModel):
    model_registry_id: str
    training_data_version: str
    retraining_trigger: bool

class Monitoring(BaseModel):
    model_version: str
    drift_score: float
    latency_ms: int
    mse_free: float

class Inference(BaseModel):
    predicted_class: Optional[str] = None
    confidence_score: Optional[float] = None
    dispersion_map_id: Optional[str] = None

class PIMLFeatures(BaseModel):
    sigma_y: Optional[float] = None
    sigma_z: Optional[float] = None
    pe_number: Optional[float] = None
    stability_index: Optional[float] = None

class ForensicEvent(BaseModel):
    simulation_id: str
    timestamp: str
    SensorAir: Any = {}
    SensorSubstance: Any = {}
    SensorGPS: Any = {}
    PIML_Features: Any = {}
    Inference: Any = {}
    Monitoring: Any = {}
    ModelOps: Any = {}
    ForensicExport: Any = {}

    @validator("timestamp")
    def ensure_iso(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", ""))
        except Exception:
            raise ValueError("timestamp must be ISO 8601")
        return v

# ============================================================
# FUNZIONI UTILI
# ============================================================

def write_bundle(event: dict) -> str:
    """
    Crea un file JSON firmato e con hash SHA-256 per ogni evento ricevuto.
    Ritorna il percorso del file creato.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"bundle_{ts}_{uuid.uuid4().hex[:8]}.json"
    path = os.path.join(FORENSIC_DIR, bundle_name)

    # Calcolo hash e firma simulata
    serialized = json.dumps(event, sort_keys=True, default=str)
    hash_value = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    signature = f"sig_{uuid.uuid4().hex[:16]}"

    bundle = {
        "timestamp": datetime.utcnow().isoformat(),
        "bundle_name": bundle_name,
        "hash_sha256": hash_value,
        "signature": signature,
        "event": event,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)

    return path


def list_bundles(n: int = 20) -> List[str]:
    files = sorted(os.listdir(FORENSIC_DIR), reverse=True)
    return files[:n]


def load_bundle(filename: str) -> dict:
    path = os.path.join(FORENSIC_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "service": "forensic_logger", "time": datetime.utcnow().isoformat()}


@app.post("/log_forensic")
def log_forensic(event: ForensicEvent):
    """
    Riceve un evento completo (es. dal servizio /ingest_data)
    e genera un forensic bundle firmato.
    """
    event_dict = json.loads(event.json(by_alias=True))

    try:
        path = write_bundle(event_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "ok",
        "bundle_saved": os.path.basename(path),
        "hash_sha256": hashlib.sha256(json.dumps(event_dict, sort_keys=True).encode()).hexdigest(),
        "path": path,
    }


@app.get("/forensic_bundles")
def get_bundles(last_n: int = Query(10, ge=1, le=100)):
    """Restituisce la lista degli ultimi N bundle presenti."""
    files = list_bundles(last_n)
    return {"status": "ok", "count": len(files), "bundles": files}


@app.get("/forensic_bundle/{filename}")
def get_bundle(filename: str):
    """Ritorna il contenuto di un bundle specifico (per verifiche forensi)."""
    try:
        data = load_bundle(filename)
        return {"status": "ok", "bundle": data}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Bundle not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/forensic_bundle/{filename}")
def delete_bundle(filename: str):
    """Elimina manualmente un bundle forense (uso di manutenzione)."""
    path = os.path.join(FORENSIC_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Bundle not found")
    os.remove(path)
    return {"status": "ok", "deleted": filename}


# ============================================================
# AVVIO LOCALE
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)
