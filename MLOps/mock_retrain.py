from fastapi import FastAPI
from datetime import datetime
import threading
import time
import json
import os
import random

# ============================================================
# MOCK RETRAIN SERVICE – Layer 5 (Feedback & ModelOps)
# ============================================================

app = FastAPI(title="MLOps Mock Retrain Service")

LOG_DIR = "/logs"
MONITOR_LOG = os.path.join(LOG_DIR, "monitoring_log.jsonl")
RETRAIN_LOG = os.path.join(LOG_DIR, "modelops_retrain_log.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

CHECK_INTERVAL = 60  # secondi tra controlli
DRIFT_THRESHOLD = 0.05  # soglia di drift medio per trigger
MIN_EVENTS = 10  # minimo eventi recenti per valutare

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
    """Simula la procedura di retraining (placeholder)."""
    new_version = f"XGBoost_v{random.randint(2, 9)}.{random.randint(0, 9)}"
    metrics = {
        "accuracy": round(random.uniform(0.82, 0.88), 3),
        "far": round(random.uniform(0.03, 0.05), 3),
        "training_samples": random.randint(2400, 2700),
        "duration_min": round(random.uniform(4.5, 6.0), 2),
    }
    return new_version, metrics


def retrain_loop():
    """Loop continuo che monitora il drift e simula retraining."""
    global running
    print(f"[RetrainService] 🔁 Avvio loop retraining (check ogni {CHECK_INTERVAL}s)")

    while running:
        time.sleep(CHECK_INTERVAL)

        events = load_recent_monitoring(50)
        if len(events) < MIN_EVENTS:
            continue

        drift_mean = compute_mean_drift(events)
        retrain_needed = drift_mean > DRIFT_THRESHOLD

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_mean": round(drift_mean, 4),
            "retrain_triggered": retrain_needed,
        }

        if retrain_needed:
            new_version, metrics = simulate_retrain()
            entry.update({
                "new_model_version": new_version,
                "metrics": metrics,
                "status": "retrained"
            })
            print(f"[RetrainService] ⚙️ Retraining simulato → Nuova versione: {new_version}")
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
