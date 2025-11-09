import datetime
import numpy as np
import pandas as pd
import os
import random
import sys

# === FIX PATHS ============================================================
# Calcola il path assoluto alla root del progetto (Pention-System)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GAUSSIAN_PATH = os.path.join(ROOT_DIR, "gaussianPuff")
sys.path.extend([ROOT_DIR, GAUSSIAN_PATH])

# Importa correttamente i moduli dal package gaussianPuff
from gaussianPuff.config import (
    ModelConfig, StabilityType, WindType, OutputType, NPS,
    PasquillGiffordStability, DispersionModelType, ConfigPuff
)
from gaussianPuff.gaussianModel import run_dispersion_model
from gaussianPuff.Sensor import SensorAir


# === PARAMETRI ============================================================
N_SIMULATIONS = 100
SAVE_DIR = "./CorrectionDispersion_PIML/dataset"
SAVE_DIR_CONC = os.path.join(SAVE_DIR, "real_dispersion")
BINARY_MAP_PATH = os.path.join(os.path.dirname(__file__), "binary_maps_data/amsterdam_netherlands_bbox.npy")
os.makedirs(SAVE_DIR_CONC, exist_ok=True)

binary_map = np.load(BINARY_MAP_PATH)
free_cells = np.argwhere(binary_map == 1)

def random_position():
    idx = np.random.choice(len(free_cells))
    y, x = free_cells[idx]
    return float(x), float(y)

def stability_index(stab):
    mapping = {
        PasquillGiffordStability.VERY_UNSTABLE: 1,
        PasquillGiffordStability.MODERATELY_UNSTABLE: 2,
        PasquillGiffordStability.SLIGHTLY_UNSTABLE: 3,
        PasquillGiffordStability.NEUTRAL: 4,
        PasquillGiffordStability.MODERATELY_STABLE: 5,
        PasquillGiffordStability.VERY_STABLE: 6
    }
    return mapping.get(stab, 0)

def sigma_from_distance(distance, stab_index):
    """Approssima σy, σz da distanza (m) e stabilità (Pasquill–Gifford)."""
    # coeff empirici
    a_y = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
    b_y = [0.90, 0.88, 0.86, 0.83, 0.80, 0.78]
    idx = max(0, min(5, stab_index - 1))
    sigma_y = a_y[idx] * (distance + 1) ** b_y[idx]
    sigma_z = 0.5 * sigma_y
    return round(float(sigma_y), 3), round(float(sigma_z), 3)

records = []

for i in range(N_SIMULATIONS):
    print(f"[SIM] {i+1}/{N_SIMULATIONS}")

    # --- Meteo e stabilità ---
    sensor_air = SensorAir(sensor_id=0, x=0.0, y=0.0, z=2.0)
    wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = sensor_air.sample_meteorology()

    # --- Sorgente ---
    x_src, y_src = random_position()
    h_src = round(np.random.uniform(1, 10), 2)
    Q = round(np.random.uniform(0.0001, 0.01), 4)
    stacks = [(x_src, y_src, Q, h_src)]

    # --- Configurazione modello ---
    disp_model = random.choice([DispersionModelType.PLUME, DispersionModelType.PUFF])
    config = ModelConfig(
        days=random.choice([5, 10, 15]),
        aerosol_type=random.choice(list(NPS)),
        dry_size=1.0,
        humidify=humidify,
        RH=RH,
        stability_profile=stability_type,
        stability_value=stability_value,
        wind_type=wind_type,
        wind_speed=wind_speed,
        output=OutputType.PLAN_VIEW,
        stacks=stacks,
        dispersion_model=disp_model,
        config_puff=ConfigPuff() if disp_model == DispersionModelType.PUFF else None
    )

    # --- Simulazione gaussiana ---
    C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = run_dispersion_model(config)

    # --- Calcolo σy, σz fisici ---
    dist = np.mean(np.sqrt((x - x_src) ** 2 + (y - y_src) ** 2))
    sigma_y, sigma_z = sigma_from_distance(dist, stability_index(stability_value))

    # --- Salvataggio mappa ---
    fname = f"sim_{i}_conc_real_{datetime.datetime.now().date()}.npy"
    np.save(os.path.join(SAVE_DIR_CONC, fname), C1)

    # --- Media direzione vento ---
    wind_angle_mean = float(np.degrees(np.arctan2(np.mean(np.sin(np.deg2rad(wind_dir))),
                                                 np.mean(np.cos(np.deg2rad(wind_dir))))))

    # --- Record simulazione ---
    records.append({
        "simulation_id": i,
        "real_concentration_name_file": fname,
        "wind_speed": wind_speed,
        "wind_dir_mean": round(wind_angle_mean, 2),
        "stability_class": stability_value.name,
        "stability_index": stability_index(stability_value),
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
        "RH": RH,
        "humidify": humidify,
        "source_x": x_src,
        "source_y": y_src,
        "source_h": h_src,
        "emission_rate": Q,
        "dispersion_model": disp_model.name
    })

# --- Salvataggio CSV aggregato ---
df = pd.DataFrame(records)
csv_path = os.path.join(SAVE_DIR, f"nps_simulated_dataset_gaussiano_{datetime.datetime.now().date()}_PIML.csv")
df.to_csv(csv_path, index=False)

print(f"\n✅ Dataset PIML generato: {csv_path}")
print(f"Totale simulazioni: {len(df)}")
