import requests
import json
import time
import numpy as np

UI = "http://localhost:8000"
INGEST = "http://localhost:8011/ingest_data"
FORENSIC = "http://localhost:8013/forensic_bundles"

sample = {
    "simulation_id": "E2E_TEST",
    "timestamp": "2025-01-01T12:00:00Z",
    "SensorAir": {
        "temperature_C": 18.0,
        "humidity_%": 0.70,
        "wind_speed_mps": 4.0,
        "wind_dir_deg": 250,
        "stability_class": "D"
    },
    "SensorSubstance": {
        "compound_name": "Cocaine",
        "spectrum_ei_1_600": [0]*600,
        "concentration_series_mg_m3": [0]*600,
        "unit": "EI_intensity",
        "noise_level": 0.05
    },
    "SensorGPS": {
        "latitude": 52.3600,
        "longitude": 4.8850,
        "altitude_m": 2.0
    },
    "SourceGPS": {
        "latitude": 52.3610,
        "longitude": 4.8860
    }
}

print("\n=== STEP 1: Sending data to /ingest_data ===")
t0 = time.time()
resp = requests.post(INGEST, json=sample).json()
t1 = time.time()
print(json.dumps(resp, indent=2))
print(f"\nLatency ingestion request: {round((t1-t0)*1000,2)} ms")
errors = []
mon = resp.get("monitoring", {})
if mon.get("latency_ms", 9999) > 500:
    errors.append("Latency too high")
if mon.get("drift_score", 1) > 0.5:
    errors.append("Unexpected high drift")

print("\n=== STEP 2: Recovering the last bundle ===")
bundles = requests.get(FORENSIC).json().get("bundles", [])
if not bundles:
    raise RuntimeError("No forensic bundles generated!")
last = bundles[0]
print("Last bundle:", last)
bundle = requests.get(f"http://localhost:8013/forensic_bundle/{last}").json()
event = bundle["bundle"]["event"]

print("\n=== STEP 3: Validations ===")
def check(cond, name):
    if cond: 
        print(f"[OK] {name}")
    else:
        print(f"[FAIL] {name}")
        errors.append(name)

check("PIML_Runtime" in event, "PIML runtime exists")
check(event["PIML_Runtime"]["correction_dispersion_piml"]["corrected_map_shape"] == [500,500], "Corrected map shape is 500x500")
px, py = event["Inference"]["predicted_source_location"]
check(np.isfinite(px) and np.isfinite(py), "Predicted source finite")
conf = event["Inference"]["confidence_score"]
check(0 <= conf <= 1, "Confidence in [0,1]")
lat = event["Monitoring"]["latency_ms"]
check(lat < 500, "Latency under limit")
drift = event["Monitoring"]["drift_score"]
check(drift < 0.6, "Drift under limit")

print("\n=== RESULT ===")
if errors:
    print("Test FAILED")
    print(errors)
else:
    print("End-to-end pipeline OK")