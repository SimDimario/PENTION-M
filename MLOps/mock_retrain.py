from fastapi import FastAPI
from datetime import datetime, timedelta
import threading
import time
import json
import os
import sys
import importlib.util

# --- Fix import manuale ---
sys.path.extend(["/MLOps", "/CorrectionDispersion_PIML", "/gaussianPuff"])

print("[RetrainService] mock_retrain.py LOADED (import container)")

# ============================================================
# Loader dinamico per il retraining reale PIML
# ============================================================

def load_piml_retrain():
    """
    Carica dinamicamente il modulo service_train_piml.py da /CorrectionDispersion_PIML.
    Deve esporre una funzione: retrain_model() -> (new_version: str, metrics: dict)
    """
    module_path = "/CorrectionDispersion_PIML/service_train_piml.py"
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Modulo non trovato: {module_path}")

    spec = importlib.util.spec_from_file_location("service_train_piml", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ============================================================
# MOCK / REAL RETRAIN SERVICE – Layer 5 (Feedback & ModelOps)
# ============================================================

app = FastAPI(title="MLOps Retrain Service")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[RetrainService] Lifespan startup")
    start_retrain_thread()
    yield
    print("[RetrainService] Lifespan shutdown")
    global running
    running = False

app.router.lifespan_context = lifespan

LOG_DIR = "/logs"
MONITOR_LOG = os.path.join(LOG_DIR, "monitoring_log.jsonl")
RETRAIN_LOG = os.path.join(LOG_DIR, "modelops_retrain_log.jsonl")
REGISTRY_PATH = os.path.join(LOG_DIR, "model_registry.json")

os.makedirs(LOG_DIR, exist_ok=True)

CHECK_INTERVAL = 30      # ogni 10s
DRIFT_THRESHOLD = 0.6    # soglia per il drift medio
DRIFT_WINDOW = 10         # numero di eventi usati per la media
MIN_EVENTS = 10           # minimo eventi per attivare logica

running = True  # flag di controllo thread
LAST_RETRAIN_AT = None  # timestamp dell’ultimo retrain riuscito

# Flag per evitare di creare più thread in caso di import multipli
AUTO_THREAD_STARTED = False

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
    Simula un retraining (usato SOLO per trigger manuale).
    Versioning coerente: PIML_v1, PIML_v2, ...
    """
    registry_path = REGISTRY_PATH
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
        "description": "Mock retrain executed due to manual trigger",
        "drift_reset": True
    }

    return new_version, metrics

def retrain_loop():
    """Loop continuo che controlla drift e decide se retrainare."""
    global running, LAST_RETRAIN_AT
    print(f"[RetrainService] Avvio loop retraining (check ogni {CHECK_INTERVAL}s)")

    while running:
        time.sleep(CHECK_INTERVAL)

        now = datetime.utcnow()

        # cooldown di 5 minuti tra due retrain
        if LAST_RETRAIN_AT is not None:
            if (now - LAST_RETRAIN_AT) < timedelta(minutes=5):
                continue

        # 1) Leggi ultimi eventi di monitoring
        events = load_recent_monitoring(DRIFT_WINDOW)
        if len(events) < MIN_EVENTS:
            # comment di debug leggero
            continue

        last_event = events[-1]
        event_model_version = last_event.get("model_version")

        # 2) Leggi versione modello dal registry
        if os.path.exists(REGISTRY_PATH):
            try:
                with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                registry_version = registry.get("current_model_version")
            except Exception:
                registry_version = None
        else:
            registry_version = None

        # 3) Calcola drift medio sulle ultime N simulazioni
        drift_mean = compute_mean_drift(events)
        drift_trigger = drift_mean > DRIFT_THRESHOLD

        # opzionale: richiedi almeno un evento negli ultimi 5 minuti
        try:
            recent = [
                e for e in events
                if datetime.fromisoformat(e["time"]) > datetime.utcnow() - timedelta(minutes=5)
            ]
        except Exception:
            recent = events

        if len(recent) == 0:
            drift_trigger = False

        mismatch_trigger = False
        retrain_needed = drift_trigger or mismatch_trigger

        if not retrain_needed:
            continue

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_mean": round(drift_mean, 4),
            "registry_version": registry_version,
            "event_model_version": event_model_version,
            "mismatch_trigger": mismatch_trigger,
            "drift_trigger": drift_trigger,
            "retrain_triggered": retrain_needed,
        }

        try:
            print("[RetrainService] Avvio retraining PIML reale...")
            piml_mod = load_piml_retrain()
            new_version, metrics = piml_mod.retrain_model()

            entry.update({
                "new_model_version": new_version,
                "metrics": metrics,
                "status": "retrained",
                "fallback_used": False,
            })

            print(f"[RetrainService] Retraining COMPLETATO → Nuova versione: {new_version}")

            # Aggiorna registry
            with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "last_update": entry["timestamp"],
                    "current_model_version": new_version,
                    "previous_model_version": registry_version,
                    "training_data_version": f"PIML_DS_v{int(new_version.replace('PIML_v',''))}",
                    "metrics": metrics
                }, f, indent=2)

            # 🔄 RESET DRIFT BASELINE DOPO RETRAIN
            drift_baseline_path = "/logs/drift_baseline.json"
            try:
                with open(drift_baseline_path, "w", encoding="utf-8") as f:
                    json.dump({"mean": None, "cov": None, "count": 0, "distances": []}, f)
                print("[RetrainService] Baseline drift resettata dopo retrain.")
            except Exception as e:
                print(f"[RetrainService] ERRORE reset baseline drift: {e}")

            # 🔄 RESET finestra per evitare drift ricorsivo
            with open(MONITOR_LOG, "w") as f:
                pass  # svuota completamente il monitoring log
            print("[RetrainService] Monitoring log svuotato dopo retrain.")


            append_log(RETRAIN_LOG, entry)
            LAST_RETRAIN_AT = now

        except Exception as e:
            entry.update({
                "status": "retrain_failed",
                "error": str(e),
                "fallback_used": False,
            })
            print(f"[RetrainService] Retraining reale FALLITO: {e}")
            append_log(RETRAIN_LOG, entry)

    print("[RetrainService] Loop retraining terminato")

# ============================================================
# FUNZIONE PER AVVIARE IL THREAD (usata sia da startup che da import)
# ============================================================

def start_retrain_thread():
    global AUTO_THREAD_STARTED
    if AUTO_THREAD_STARTED:
        return
    AUTO_THREAD_STARTED = True
    print("[RetrainService] Avvio thread di background per il retrain...")
    thread = threading.Thread(target=retrain_loop, daemon=True)
    thread.start()

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
    """
    Permette di forzare un retraining MOCK manuale.
    """
    new_version, metrics = simulate_retrain()
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "manual_trigger": True,
        "new_model_version": new_version,
        "metrics": metrics,
        "status": "manual_mock_retrained"
    }
    append_log(RETRAIN_LOG, entry)
    print(f"[RetrainService] Retraining manuale (MOCK) → {new_version}")

    # Aggiorniamo comunque il registry per il test manuale
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = {}
    else:
        registry = {}

    registry.update({
        "last_update": entry["timestamp"],
        "previous_model_version": registry.get("current_model_version"),
        "current_model_version": new_version,
        "training_data_version": f"PIML_DS_v{int(new_version.replace('PIML_v',''))}",
        "metrics": metrics
    })

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    return {"status": "ok", "message": "Manual MOCK retrain completed", "new_version": new_version}


if os.environ.get("MLOPS_AUTOSTART_RETRAIN", "1") == "1":
    start_retrain_thread()

# ============================================================
# ESECUZIONE LOCALE
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8014)
