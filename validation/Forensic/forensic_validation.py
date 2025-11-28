import json
import os
import hashlib
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


# ============
#  PATHS
# ============

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLE_DIR = os.path.join(BASE_DIR, "logs", "forensic")

MODEL_PATH = os.path.join(BASE_DIR, "CorrectionDispersion_PIML", "models", "mcxm_piml_model_best.pth")
MAP_DIR = os.path.join(BASE_DIR, "CorrectionDispersion_PIML", "dataset", "real_dispersion")


# ============
#  HASH UTILS
# ============

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============
#  CANONICAL JSON (replica identica al forensic_logger)
# ============

def canonicalize_event(event: dict) -> bytes:
    """
    Produce esattamente la stessa canonicalizzazione del forensic_logger:
    1. json.dumps(event, sort_keys=True, default=str)
    2. json.loads(...)
    3. json.dumps(..., sort_keys=True, default=str)
    """

    tmp = json.dumps(event, sort_keys=True, default=str)
    obj = json.loads(tmp)
    serialized = json.dumps(obj, sort_keys=True, default=str)
    return serialized.encode("utf-8")


# ============
#  VALIDATION LOGIC
# ============

def load_bundle(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_signature(hash_value: str, signature_hex: str, public_key_pem: str) -> bool:
    try:
        pub: Ed25519PublicKey = serialization.load_pem_public_key(public_key_pem.encode())
        pub.verify(bytes.fromhex(signature_hex), hash_value.encode())
        return True
    except Exception:
        return False


def verify_artifacts(event):
    artifacts = event.get("artifacts", {})
    results = {}

    # model hash
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            results["model_hash_match"] = (sha256_bytes(f.read()) == artifacts.get("model_hash"))
    else:
        results["model_hash_match"] = False

    # map hash
    if os.path.exists(MAP_DIR):
        maps = sorted([f for f in os.listdir(MAP_DIR) if f.endswith(".npy")])
        if maps:
            latest_map = os.path.join(MAP_DIR, maps[-1])
            with open(latest_map, "rb") as f:
                results["map_hash_match"] = (sha256_bytes(f.read()) == artifacts.get("concentration_map_hash"))
        else:
            results["map_hash_match"] = False
    else:
        results["map_hash_match"] = False

    return results


def validate_bundle(path):
    print(f"\n=== VALIDATING BUNDLE: {path} ===")

    bundle = load_bundle(path)
    event = bundle["event"]

    expected_hash = bundle["hash_sha256"]
    signature = bundle["signature"]
    public_key = bundle["public_key"]

    # --- canonical JSON exactly as forensic_logger ---
    canonical = canonicalize_event(event)

    recomputed_hash = sha256_bytes(canonical)

    hash_match = (recomputed_hash == expected_hash)
    signature_ok = verify_signature(recomputed_hash, signature, public_key)
    artifact_results = verify_artifacts(event)

    result = {
        "hash_match": hash_match,
        "signature_ok": signature_ok,
        **artifact_results,
    }

    print(json.dumps(result, indent=2))
    return result


# ============
#  MAIN
# ============

if __name__ == "__main__":
    files = sorted([f for f in os.listdir(BUNDLE_DIR) if f.endswith(".json")])

    for f in files:
        validate_bundle(os.path.join(BUNDLE_DIR, f))
