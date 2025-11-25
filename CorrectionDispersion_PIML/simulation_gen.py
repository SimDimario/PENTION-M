import datetime
import numpy as np
import pandas as pd
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# === FIX PATHS ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GAUSSIAN_PATH = os.path.join(ROOT_DIR, "gaussianPuff")
sys.path.extend([ROOT_DIR, GAUSSIAN_PATH])

from gaussianPuff.config import (
    ModelConfig, StabilityType, WindType, OutputType, NPS,
    PasquillGiffordStability, DispersionModelType, ConfigPuff
)
from gaussianPuff.gaussianModel import run_dispersion_model
from gaussianPuff.Sensor import SensorAir

# === PARAMETRI ============================================================
N_SIMULATIONS = 1000
SAVE_EVERY = 100
KEEP_NPY_UNTIL = 300
N_WORKERS = 8   # ⬅️ usa 8 processi in parallelo (puoi aumentare fino a 12–14)
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
    a_y = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
    b_y = [0.90, 0.88, 0.86, 0.83, 0.80, 0.78]
    idx = max(0, min(5, stab_index - 1))
    sigma_y = a_y[idx] * (distance + 1) ** b_y[idx]
    sigma_z = 0.5 * sigma_y
    return round(float(sigma_y), 3), round(float(sigma_z), 3)


# === FUNZIONE PARALLELA ===================================================
def run_single_simulation(i, csv_date, keep_until=KEEP_NPY_UNTIL):
    sensor_air = SensorAir(sensor_id=0, x=0.0, y=0.0, z=2.0)
    wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = sensor_air.sample_meteorology()

    # 1. posizione sorgente nella binary map
    x_src, y_src = random_position()

    # 2. converti pixel → metri (pixel * 10m - 2500m per centrare)
    px_to_m = 10.0
    x_src_m = x_src * px_to_m - 2500.0
    y_src_m = y_src * px_to_m - 2500.0

    # 3. altezza e Q fisici
    h_src = round(np.random.uniform(1, 10), 2)
    Q = np.random.uniform(5.0, 50.0) / 3600.0  # kg/s

    # 4. stacks corretti
    stacks = [(x_src_m, y_src_m, Q, h_src)]


    disp_model = DispersionModelType.PLUME
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
        dispersion_model=DispersionModelType.PLUME,
        config_puff=None
    )

    C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = run_dispersion_model(config)

    fname = f"sim_{i}_conc_real_{csv_date}.npy"

    # Salviamo SOLO per le prime `keep_until` simulazioni
    # e SOLO la snapshot 2D finale (niente tensore 3D gigante)
    if i < keep_until:
        conc_path = os.path.join(SAVE_DIR_CONC, fname)
        conc_2d = C1[:, :, -1].astype(np.float32)  # snapshot finale, 2D, molto leggera
        np.save(conc_path, conc_2d)
    # per i >= keep_until NON salviamo proprio il file .npy
    # ma continuiamo a restituire fname nel dizionario (serve solo al CSV)

    wind_angle_mean = float(np.degrees(np.arctan2(np.mean(np.sin(np.deg2rad(wind_dir))),
                                                 np.mean(np.cos(np.deg2rad(wind_dir))))))

    return {
        "simulation_id": i,
        "real_concentration_name_file": fname,
        "wind_speed": wind_speed,
        "wind_dir_mean": round(wind_angle_mean, 2),
        "stability_class": stability_value.name,
        "stability_index": stability_index(stability_value),
        "RH": RH,
        "humidify": humidify,
        "source_x": x_src,
        "source_y": y_src,
        "source_h": h_src,
        "emission_rate": Q,
        "dispersion_model": disp_model.name
    }

# === MAIN LOOP PARALLELIZZATO ============================================
if __name__ == "__main__":
    csv_date = datetime.datetime.now().date()
    temp_csv_path = os.path.join(SAVE_DIR, f"nps_simulated_dataset_gaussiano_{csv_date}_PIML_partial.csv")

    results = []

    print(f"🚀 Avvio simulazioni con {N_WORKERS} processi paralleli...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        # Usa solo executor.map → più efficiente e stabile su Windows
        for result in tqdm(
            executor.map(run_single_simulation, range(N_SIMULATIONS), [csv_date] * N_SIMULATIONS),
            total=N_SIMULATIONS,
            desc="Simulazioni in corso",
            dynamic_ncols=True
        ):
            results.append(result)

            # Checkpoint ogni SAVE_EVERY simulazioni
            if len(results) % SAVE_EVERY == 0:
                df_partial = pd.DataFrame(results)
                df_partial["gps_x"] = df_partial["source_x"] / df_partial["source_x"].max()
                df_partial["gps_y"] = df_partial["source_y"] / df_partial["source_y"].max()
                df_partial.to_csv(temp_csv_path, index=False)
                print(f"\n💾 Checkpoint salvato ({len(results)}/{N_SIMULATIONS}) → {temp_csv_path}")
                sys.stdout.flush()  # forza aggiornamento output

    # === SALVATAGGIO FINALE =============================================
    df = pd.DataFrame(results)
    df["gps_x"] = df["source_x"] / df["source_x"].max()
    df["gps_y"] = df["source_y"] / df["source_y"].max()

    csv_path = os.path.join(SAVE_DIR, f"nps_simulated_dataset_gaussiano_{csv_date}_PIML.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Dataset PIML completo generato: {csv_path}")
    print(f"Totale simulazioni: {len(df)}")
    print(f"📊 GPS normalizzato incluso nel dataset.")
    print(f"💾 Prime {KEEP_NPY_UNTIL} mappe salvate in: {SAVE_DIR_CONC}")
    print(f"⚡ Le restanti sono state eliminate per risparmiare spazio.")
