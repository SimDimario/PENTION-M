import requests
import json
import time
import os
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
INGEST_URL = "http://localhost:8011/ingest_data"
MONITOR_URL = "http://localhost:8012/metrics/summary"
FORENSIC_URL = "http://localhost:8013/forensic_bundles"
RETRAIN_URL = "http://localhost:8014/health"

LOG_DIR = "./logs"
FORENSIC_DIR = os.path.join(LOG_DIR, "forensic")

# =========================================================
# PAYLOAD DI TEST (simulazione realistica)
# =========================================================
payload = {
    "simulation_id": "sim_test_001",
    "timestamp": datetime.utcnow().isoformat() + "Z",

    "SensorAir": {
        "temperature_C": 21.3,
        "humidity_%": 0.62,
        "wind_speed_mps": 4.8,
        "wind_dir_deg": 270,
        "stability_class": "NEUTRAL"
    },

    "SensorSubstance": {
        "compound_name": "Cathinone",
        "molecular_formula": "C9H11NO",
        "concentration_series_mg_m3": [0.002, 0.003, 0.0025, 0.0018],
        "unit": "mg/m3",
        "noise_level": 0.05
    },

    "SensorGPS": {
        "latitude": 52.3702,
        "longitude": 4.8952,
        "altitude_m": 2.0
    },

    "PIML_Features": {
        "sigma_y": 0.12,
        "sigma_z": 0.06,
        "pe_number": 0.9,
        "wind_vector": [5.0, 0.0],
        "stability_index": 4
    },

    "Inference": {
        "dispersion_map_id": "map_123",
        "predicted_source_location": [45.0, 12.0],
        "predicted_class": "Cathinones",
        "confidence_score": 0.92
    },

    "Monitoring": {
        "model_version": "v1.0.0",
        "drift_score": 0.05,
        "latency_ms": 120,
        "mse_free": 0.00023
    },

    "ModelOps": {
        "model_registry_id": "mdl_v1_2025",
        "training_data_version": "PIML_DS_v1",
        "retraining_trigger": True
    },

    "UI_Output": {
        "dashboard_tabs": ["Dispersion", "Source Localization", "NPS Classification"],
        "visualization_files": ["dispersion_map.html", "source_plot.png"]
    },

    "ForensicExport": {
        "export_file": "test_bundle.json",
        "hash_sha256": "dummyhash",
        "signature": "sig_test",
        "compliance_tags": ["ISO27001", "GDPR"]
    }
}

# =========================================================
# STEP 1 – POST verso ingestion
# =========================================================
print("🚀 Step 1: Invio payload a /ingest_data ...")
r = requests.post(INGEST_URL, json=payload)
print("Status:", r.status_code)
print("Response:", r.text[:300], "...\n")

# =========================================================
# STEP 2 – Attesa propagazione log
# =========================================================
print("⏳ Step 2: Attesa 3 secondi per propagazione log...")
time.sleep(3)

# =========================================================
# STEP 3 – Verifica monitoring
# =========================================================
print("📊 Step 3: Lettura riepilogo da /metrics/summary ...")
r = requests.get(MONITOR_URL)
print("Status:", r.status_code)
print("Response:", json.dumps(r.json(), indent=2)[:500], "...\n")

# =========================================================
# STEP 4 – Verifica forensic bundle
# =========================================================
print("🧾 Step 4: Lettura ultimi bundle forensi ...")
r = requests.get(FORENSIC_URL)
print("Status:", r.status_code)
if r.ok:
    bundles = r.json().get("bundles", [])
    print("Found bundles:", len(bundles))
    if bundles:
        print("Ultimo bundle:", bundles[0])
else:
    print("Errore forensic:", r.text)

# =========================================================
# STEP 5 – Verifica retrain health
# =========================================================
print("\n🔁 Step 5: Check retrain service ...")
r = requests.get(RETRAIN_URL)
print("Status:", r.status_code)
print("Response:", r.text)
print("\n✅ Test completato.")
