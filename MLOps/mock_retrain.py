from fastapi import FastAPI
from datetime import datetime
import threading
import time
import json
import os
import random
import sys

# --- Fix import manuale ---
sys.path.extend(["/MLOps", "/CorrectionDispersion_PIML", "/gaussianPuff"])

import importlib.util

# --- Loader dinamico per il retraining reale PIML ---
def load_piml_retrain():
    """Carica dinamicamente il modulo service_train_piml.py da /CorrectionDispersion_PIML"""
    module_path = "/CorrectionDispersion_PIML/service_train_piml.py"
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Modulo non trovato: {module_path}")
    spec = importlib.util.spec_from_file_location("service_train_piml", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ============================================================
# MOCK RETRAIN SERVICE – Layer 5 (Feedback & ModelOps)
# ============================================================

app = FastAPI(title="MLOps Mock Retrain Service")

LOG_DIR = "/logs"
MONITOR_LOG = os.path.join(LOG_DIR, "monitoring_log.jsonl")
RETRAIN_LOG = os.path.join(LOG_DIR, "modelops_retrain_log.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

CHECK_INTERVAL = 10     # controlla ogni 10 secondi (solo per test)
DRIFT_THRESHOLD = 0.12  # soglia media più realistica
MIN_EVENTS = 5          # bastano pochi eventi

running = True  # flag di controllo thread


# ============================================================
# FUNZIONI UTILI
# ============================================================

def load_recent_monitoring(n: int = 50):
    """Carica le ultime N righe del monitoring log."""
    if not os.path.exists(MONITOR_LOG):
        return []
    with open(MONITOR_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    data = []
    for line in lines:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return data

def append_log(path: str, entry: dict):
    """Aggiunge una riga JSONL a un log file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

def compute_mean_drift(events):
    """Calcola il drift medio sugli ultimi eventi."""
    vals = [e.get("drift_score") for e in events if isinstance(e.get("drift_score"), (float, int))]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)

def simulate_retrain():
    """
    Simula un retraining quando il retrain reale fallisce.
    Versioning coerente: PIML_v1, PIML_v2, ...
    """
    registry_path = os.path.join(LOG_DIR, "model_registry.json")
    current = 0

    # Leggi versione attuale dal registry
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                v = data.get("current_model_version", "")
                if v.startswith("PIML_v"):
                    current = int(v.replace("PIML_v", ""))
        except Exception:
            pass

    # Incrementa versione
    new_version = f"PIML_v{current + 1}"

    # Metriche fittizie coerenti
    metrics = {
        "description": "Mock retrain executed due to fallback",
        "drift_reset": True
    }

    return new_version, metrics

def read_previous_model_version():
    """Legge la versione precedente del modello dal registry, se presente."""
    registry_path = os.path.join(LOG_DIR, "model_registry.json")
    if not os.path.exists(registry_path):
        return None
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("current_model_version")
    except Exception:
        return None

def retrain_loop():
    """Loop continuo che controlla drift, version mismatch e trigger vari."""
    global running
    print(f"[RetrainService] 🔁 Avvio loop retraining (check ogni {CHECK_INTERVAL}s)")

    registry_path = os.path.join(LOG_DIR, "model_registry.json")

    while running:
        time.sleep(CHECK_INTERVAL)

        # -------------------------------------------------------
        # 1️⃣ CARICA EVENTI DI MONITORING
        # -------------------------------------------------------
        events = load_recent_monitoring(50)
        if len(events) < MIN_EVENTS:
            continue

        # ultimo evento
        last_event = events[-1]
        event_model_version = last_event.get("model_version")

        # -------------------------------------------------------
        # 2️⃣ CARICA MODEL VERSION DAL REGISTRY
        # -------------------------------------------------------
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
            registry_version = registry.get("current_model_version")
        else:
            registry_version = None

        # -------------------------------------------------------
        # 3️⃣ CHECK DRIFT
        # -------------------------------------------------------
        drift_mean = compute_mean_drift(events)
        drift_trigger = drift_mean > DRIFT_THRESHOLD

        # -------------------------------------------------------
        # 4️⃣ CHECK VERSION MISMATCH
        # -------------------------------------------------------
        mismatch_trigger = (
            registry_version is not None and
            event_model_version is not None and
            event_model_version != registry_version
        )

        # Flag finale
        retrain_needed = drift_trigger or mismatch_trigger

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_mean": round(drift_mean, 4),
            "registry_version": registry_version,
            "event_model_version": event_model_version,
            "mismatch_trigger": mismatch_trigger,
            "drift_trigger": drift_trigger,
            "retrain_triggered": retrain_needed,
        }

        # -------------------------------------------------------
        # 5️⃣ SE SERVE → ESEGUI RETRAIN
        # -------------------------------------------------------
        if retrain_needed:
            try:
                piml_mod = load_piml_retrain()
                new_version, metrics = piml_mod.retrain_model()
                # Versioning coerente: converte PIML_vHHMMSS in versione incrementale
                try:
                    current = 0
                    if registry_version and registry_version.startswith("PIML_v"):
                        current = int(registry_version.replace("PIML_v", ""))
                    new_version = f"PIML_v{current + 1}"
                except Exception:
                    new_version = "PIML_v1"
            except Exception as e:
                print(f"[RetrainService] ⚠️ Retraining reale fallito: {e}")
                new_version, metrics = simulate_retrain()

            entry.update({
                "new_model_version": new_version,
                "metrics": metrics,
                "status": "retrained",
            })

            print(f"[RetrainService] ⚙️ Retraining → Nuova versione: {new_version}")

            # AGGIORNA REGISTRY
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump({
                    "last_update": entry["timestamp"],
                    "current_model_version": new_version,
                    "previous_model_version": registry_version,
                    "metrics": metrics
                }, f, indent=2)

        else:
            entry["status"] = "stable"

        append_log(RETRAIN_LOG, entry)

    print("[RetrainService] 🛑 Loop terminato")

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "service": "mock_retrain", "time": datetime.utcnow().isoformat()}


@app.get("/status")
def get_status():
    """Ritorna l’ultimo stato del retrain log."""
    if not os.path.exists(RETRAIN_LOG):
        return {"status": "ok", "entries": 0}
    with open(RETRAIN_LOG, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if not lines:
        return {"status": "ok", "entries": 0}
    last = lines[-1]
    return {"status": "ok", "last_entry": last, "entries": len(lines)}


@app.post("/trigger_retrain")
def trigger_manual():
    """Permette di forzare un retraining manuale."""
    new_version, metrics = simulate_retrain()
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "manual_trigger": True,
        "new_model_version": new_version,
        "metrics": metrics,
        "status": "manual_retrained"
    }
    append_log(RETRAIN_LOG, entry)
    print(f"[RetrainService] 🧠 Retraining manuale → {new_version}")
    return {"status": "ok", "message": "Manual retrain completed", "new_version": new_version}

# ============================================================
# THREAD DI BACKGROUND
# ============================================================

@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=retrain_loop, daemon=True)
    thread.start()


@app.on_event("shutdown")
def shutdown_event():
    global running
    running = False


# ============================================================
# ESECUZIONE LOCALE
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8014)
