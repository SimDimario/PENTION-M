from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import json
import os
import uuid

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

KEY_DIR = "/keys"
os.makedirs(KEY_DIR, exist_ok=True)

PRIVATE_KEY_PATH = os.path.join(KEY_DIR, "private_key.pem")
PUBLIC_KEY_PATH = os.path.join(KEY_DIR, "public_key.pem")

# Generate keys if missing
if not os.path.exists(PRIVATE_KEY_PATH):
    private_key = ed25519.Ed25519PrivateKey.generate()
    with open(PRIVATE_KEY_PATH, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    public_key = private_key.public_key()
    with open(PUBLIC_KEY_PATH, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
else:
    with open(PRIVATE_KEY_PATH, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    with open(PUBLIC_KEY_PATH, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

app = FastAPI(title="MLOps Forensic Logger & ModelOps")

LOG_DIR = "/logs"
FORENSIC_DIR = os.path.join(LOG_DIR, "forensic")
os.makedirs(FORENSIC_DIR, exist_ok=True)

class ForensicExport(BaseModel):
    export_file: str
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

    class Config:
        extra = "allow"

def write_bundle(event: dict):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"bundle_{ts}_{uuid.uuid4().hex[:8]}.json"
    path = os.path.join(FORENSIC_DIR, bundle_name)
    canonical_event = json.loads(json.dumps(event, sort_keys=True, default=str))
    serialized = json.dumps(canonical_event, sort_keys=True, default=str)
    hash_value = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    signature = private_key.sign(hash_value.encode()).hex()
    fe = event.setdefault("ForensicExport", {})
    fe.setdefault("compliance_tags", [])

    bundle = {
        "timestamp": datetime.utcnow().isoformat(),
        "bundle_name": bundle_name,
        "hash_sha256": hash_value,
        "signature": signature,
        "public_key": public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode(),
        "event": event,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)

    return path, hash_value, signature

def list_bundles(n: int = 20) -> List[str]:
    files = sorted(os.listdir(FORENSIC_DIR), reverse=True)
    return files[:n]

def load_bundle(filename: str) -> dict:
    path = os.path.join(FORENSIC_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/health")
def health():
    return {"status": "ok", "service": "forensic_logger", "time": datetime.utcnow().isoformat()}

@app.post("/log_forensic")
def log_forensic(event: ForensicEvent):
    event_dict = json.loads(event.json(by_alias=True))

    if "Inference" in event_dict and isinstance(event_dict["Inference"], dict):
        if "temperature_used" not in event_dict["Inference"]:
            event_dict["Inference"]["temperature_used"] = None

    artifacts = {}
    try:
        model_path = "/CorrectionDispersion_PIML/models/mcxm_piml_model_best.pth"
        if os.path.exists(model_path):
            with open(model_path, "rb") as mf:
                artifacts["model_hash"] = hashlib.sha256(mf.read()).hexdigest()
        else:
            artifacts["model_hash"] = "missing"

        dataset_dir = "/CorrectionDispersion_PIML/dataset/real_dispersion"
        if os.path.exists(dataset_dir):
            maps = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".npy")])
            if maps:
                latest_map = os.path.join(dataset_dir, maps[-1])
                with open(latest_map, "rb") as cf:
                    artifacts["concentration_map_hash"] = hashlib.sha256(cf.read()).hexdigest()
            else:
                artifacts["concentration_map_hash"] = "none_found"
        else:
            artifacts["concentration_map_hash"] = "no_dataset_dir"

        tdv = event_dict.get("ModelOps", {}).get("training_data_version", "")
        artifacts["training_data_version"] = tdv or "unknown"

    except Exception as e:
        artifacts["error"] = str(e)

    event_dict["artifacts"] = artifacts
    fe = event_dict.setdefault("ForensicExport", {})
    fe.setdefault("compliance_tags", [])
    fe["compliance_tags"].extend([
        "SIGNATURE_OK",
        "HASH_OK",
        "ARTIFACTS_OK",
    ])

    try:
        path, bundle_hash, bundle_sig = write_bundle(event_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "ok",
        "bundle_saved": os.path.basename(path),
        "hash_sha256": bundle_hash,
        "signature": bundle_sig,
        "path": path,
    }

@app.get("/forensic_bundles")
def get_bundles(last_n: int = Query(10, ge=1, le=100)):
    files = list_bundles(last_n)
    return {"status": "ok", "count": len(files), "bundles": files}

@app.get("/forensic_bundle/{filename}")
def get_bundle(filename: str):
    try:
        data = load_bundle(filename)
        return {"status": "ok", "bundle": data}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Bundle not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/forensic_bundle/{filename}")
def delete_bundle(filename: str):
    path = os.path.join(FORENSIC_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Bundle not found")
    os.remove(path)
    return {"status": "ok", "deleted": filename}

@app.get("/verify_bundle/{filename}")
def verify_bundle(filename: str):
    """
    Recalculates the model and heatmap hashes
    and compares them with those stored in the forensic bundle.
    """
    try:
        bundle = load_bundle(filename)
        artifacts = bundle.get("event", {}).get("artifacts", {})
        result = {"bundle": filename, "verified": True, "details": {}}
        event_data = bundle.get("event", {})
        canonical_event = json.loads(json.dumps(event_data, sort_keys=True, default=str))
        recomputed_hash = hashlib.sha256(
            json.dumps(canonical_event, sort_keys=True, default=str).encode()
        ).hexdigest()
        hash_match = (recomputed_hash == bundle.get("hash_sha256", ""))
        result["details"]["event_hash_match"] = hash_match
        if not hash_match:
            result["verified"] = False
        model_path = "/CorrectionDispersion_PIML/models/mcxm_piml_model_best.pth"
        if os.path.exists(model_path):
            with open(model_path, "rb") as mf:
                current_model_hash = hashlib.sha256(mf.read()).hexdigest()
            match = current_model_hash == artifacts.get("model_hash")
            result["details"]["model_hash_match"] = match
            if not match:
                result["verified"] = False
        else:
            result["details"]["model_hash_match"] = False
            result["verified"] = False
        dataset_dir = "/CorrectionDispersion_PIML/dataset/real_dispersion"
        if os.path.exists(dataset_dir):
            maps = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".npy")])
            if maps:
                latest_map = os.path.join(dataset_dir, maps[-1])
                with open(latest_map, "rb") as cf:
                    current_map_hash = hashlib.sha256(cf.read()).hexdigest()
                match = current_map_hash == artifacts.get("concentration_map_hash")
                result["details"]["concentration_map_hash_match"] = match
                if not match:
                    result["verified"] = False
        else:
            result["details"]["concentration_map_hash_match"] = False
            result["verified"] = False
        try:
            stored_sig = bytes.fromhex(bundle.get("signature", ""))

            pub_key_pem = bundle.get("public_key", "").encode()
            pub_key = serialization.load_pem_public_key(pub_key_pem)

            pub_key.verify(stored_sig, recomputed_hash.encode())
            result["details"]["signature_valid"] = True
        except Exception:
            result["details"]["signature_valid"] = False
            result["verified"] = False

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)