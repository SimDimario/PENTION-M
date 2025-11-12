import random
import uuid
import numpy as np
from datetime import datetime

# ============================================================
# SENSOR SIMULATOR FOR PENTION-M (Layer 0 – Edge I/O Simulated)
# ============================================================

class SensorSimM:
    """
    Simulatore di sensori per PENTION-M.
    Genera dati meteorologici, GPS e di sostanza
    coerenti con il formato sample_simulation_data.json.
    """

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------
    # SensorAir (meteorologia)
    # ------------------------------
    def _simulate_sensor_air(self) -> dict:
        stability_classes = ["A", "B", "C", "D", "E", "F"]
        stability = random.choice(stability_classes)

        wind_speed = round(random.uniform(0.5, 8.0), 2)
        wind_dir_deg = random.randint(0, 360)
        RH = round(random.uniform(20, 95), 1)
        temperature = round(random.uniform(10, 30), 1)

        return {
            "temperature_C": temperature,
            "humidity_%": RH,
            "wind_speed_mps": wind_speed,
            "wind_dir_deg": wind_dir_deg,
            "stability_class": stability
        }

    # ------------------------------
    # SensorSubstance (concentrazione sintetica)
    # ------------------------------
    def _simulate_sensor_substance(self) -> dict:
        compounds = ["Cathinone", "Cannabinoid", "Phenethylamine", "Opioid", "Benzodiazepine"]
        compound = random.choice(compounds)

        t = np.linspace(0, 1, 5)
        base = np.exp(-5 * t) * np.sin(2 * np.pi * t)
        conc = np.abs(base + np.random.normal(0, 0.02, len(t)))
        noise_level = round(random.uniform(0.005, 0.02), 3)

        return {
            "compound_name": compound,
            "molecular_formula": "",
            "concentration_series_mg_m3": [round(float(v), 4) for v in conc],
            "unit": "mg/m³",
            "noise_level": noise_level
        }

    # ------------------------------
    # SensorGPS (posizione simulata)
    # ------------------------------
    def _simulate_sensor_gps(self) -> dict:
        lat = round(random.uniform(51.15, 51.25), 4)   # Amsterdam area
        lon = round(random.uniform(5.90, 6.05), 4)
        alt = round(random.uniform(0.5, 3.0), 2)
        return {"latitude": lat, "longitude": lon, "altitude_m": alt}

    # ------------------------------
    # PIML Feature layer simulato
    # ------------------------------
    def _simulate_piml_features(self, wind_speed: float, stability: str) -> dict:
        sigma_y = round(0.2 * wind_speed, 3)
        sigma_z = round(0.15 * wind_speed, 3)
        pe_number = round(wind_speed / max(0.1, np.random.uniform(0.5, 1.5)), 2)
        stability_index = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}.get(stability, 4)

        return {
            "sigma_y": sigma_y,
            "sigma_z": sigma_z,
            "pe_number": pe_number,
            "wind_vector": [wind_speed, random.randint(0, 360)],
            "stability_index": stability_index
        }

    # ------------------------------
    # Inference & Monitoring placeholder
    # ------------------------------
    def _simulate_inference_block(self, compound: str) -> dict:
        confidence = round(random.uniform(0.85, 0.99), 2)
        return {
            "dispersion_map_id": f"MAP_{uuid.uuid4().hex[:6]}.npy",
            "predicted_source_location": [random.randint(50, 150), random.randint(50, 150)],
            "predicted_class": compound,
            "confidence_score": confidence
        }

    def _simulate_monitoring_block(self) -> dict:
        return {
            "model_version": "XGBoost_v1.2",
            "drift_score": round(random.uniform(0.0, 0.05), 3),
            "latency_ms": random.randint(300, 900),
            "mse_free": round(random.uniform(0.002, 0.01), 4)
        }

    # ------------------------------
    # Generatore principale
    # ------------------------------
    def generate_simulation(self) -> dict:
        sim_id = f"SIM_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        air = self._simulate_sensor_air()
        substance = self._simulate_sensor_substance()
        gps = self._simulate_sensor_gps()
        piml_feat = self._simulate_piml_features(air["wind_speed_mps"], air["stability_class"])
        inference = self._simulate_inference_block(substance["compound_name"])
        monitoring = self._simulate_monitoring_block()

        return {
            "simulation_id": sim_id,
            "timestamp": datetime.utcnow().isoformat(),
            "SensorAir": air,
            "SensorSubstance": substance,
            "SensorGPS": gps,
            "PIML_Features": piml_feat,
            "Inference": inference,
            "Monitoring": monitoring,
            "ModelOps": {
                "model_registry_id": "mdl_XGB_2025_11",
                "training_data_version": "PENTION_EI_Complete_v3",
                "retraining_trigger": False
            },
            "UI_Output": {
                "dashboard_tabs": [
                    "Simulation", "Dispersion", "Source", "NPS", "MLOps Monitoring"
                ],
                "visualization_files": ["dispersion_map.html", "wind_rose.png"]
            },
            "ForensicExport": {
                "export_file": f"{sim_id}_bundle.zip",
                "hash_sha256": uuid.uuid4().hex,
                "signature": "sig_" + uuid.uuid4().hex[:16],
                "compliance_tags": ["GDPR", "LEA_audit_ok"]
            }
        }
    
# ============================================================
# Sensor Network Generator (STEP 2)
# ============================================================

def generate_sensor_network_from_map(conc_map: np.ndarray,
                                     building_map: np.ndarray,
                                     n_sensors: int = 5,
                                     fault_rate: float = 0.1,
                                     seed: int | None = None):
    """
    Genera una rete di sensori fisici a partire da una mappa di concentrazione.
    - Evita celle occupate (building_map == 1)
    - Campiona concentrazione reale
    - Aggiunge rumore e fault_rate
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = conc_map.shape
    building_mask = (np.array(building_map) > 0).astype(bool)

    sensors = []
    attempts = 0
    while len(sensors) < n_sensors and attempts < n_sensors * 10:
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        if building_mask[y, x]:
            attempts += 1
            continue

        conc_value = conc_map[y, x] + np.random.normal(0, 0.01)
        is_fault = np.random.rand() < fault_rate

        sensors.append({
            "sensor_id": len(sensors) + 1,
            "sensor_is_fault": is_fault,
            "time": 0.0,
            "conc": float(np.clip(conc_value, 0, 1)),
            "wind_dir_x": 0.0,
            "wind_dir_y": 0.0,
            "wind_speed": 0.0,
            "wind_type": 1,
            "gps_x": x,
            "gps_y": y,
            "stability_value": 4.0
        })
        attempts += 1

    return sensors

# ============================================================
# Test manuale
# ============================================================
if __name__ == "__main__":
    sim = SensorSimM(seed=42)
    sample = sim.generate_simulation()
    import json
    print(json.dumps(sample, indent=2))