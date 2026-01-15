import datetime
import numpy as np
import pandas as pd
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GAUSSIAN_PATH = os.path.join(ROOT_DIR, "gaussianPuff")
sys.path.extend([ROOT_DIR, GAUSSIAN_PATH])

from gaussianPuff.config import (
    ModelConfig, StabilityType, WindType, OutputType, NPS,
    PasquillGiffordStability, DispersionModelType, ConfigPuff
)
from gaussianPuff.gaussianModel import run_dispersion_model
from gaussianPuff.Sensor import SensorAir

N_SIMULATIONS = 1000
SAVE_EVERY = 100
KEEP_NPY_UNTIL = 300
N_WORKERS = 8
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

def run_single_simulation(i, csv_date, keep_until=KEEP_NPY_UNTIL):
    sensor_air = SensorAir(sensor_id=0, x=0.0, y=0.0, z=2.0)
    wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = sensor_air.sample_meteorology()
    x_src, y_src = random_position()
    px_to_m = 10.0
    x_src_m = x_src * px_to_m - 2500.0
    y_src_m = y_src * px_to_m - 2500.0
    h_src = round(np.random.uniform(1, 10), 2)
    Q = np.random.uniform(5.0, 50.0) / 3600.0
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
    if i < keep_until:
        conc_path = os.path.join(SAVE_DIR_CONC, fname)
        conc_2d = C1[:, :, -1].astype(np.float32)
        np.save(conc_path, conc_2d)
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

if __name__ == "__main__":
    csv_date = datetime.datetime.now().date()
    temp_csv_path = os.path.join(SAVE_DIR, f"nps_simulated_dataset_gaussiano_{csv_date}_PIML_partial.csv")

    results = []

    print(f"Running simulations with {N_WORKERS} parallel processes...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        for result in tqdm(
            executor.map(run_single_simulation, range(N_SIMULATIONS), [csv_date] * N_SIMULATIONS),
            total=N_SIMULATIONS,
            desc="Simulazioni in corso",
            dynamic_ncols=True
        ):
            results.append(result)
            if len(results) % SAVE_EVERY == 0:
                df_partial = pd.DataFrame(results)
                df_partial["gps_x"] = df_partial["source_x"] / df_partial["source_x"].max()
                df_partial["gps_y"] = df_partial["source_y"] / df_partial["source_y"].max()
                df_partial.to_csv(temp_csv_path, index=False)
                print(f"\nCheckpoint saved ({len(results)}/{N_SIMULATIONS}) → {temp_csv_path}")
                sys.stdout.flush()

    df = pd.DataFrame(results)
    df["gps_x"] = df["source_x"] / df["source_x"].max()
    df["gps_y"] = df["source_y"] / df["source_y"].max()

    csv_path = os.path.join(SAVE_DIR, f"nps_simulated_dataset_gaussiano_{csv_date}_PIML.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nComplete PIML dataset generated: {csv_path}")
    print(f"Total simulations: {len(df)}")
    print(f"Normalized GPS included in the dataset.")
    print(f"First {KEEP_NPY_UNTIL} maps saved in: {SAVE_DIR_CONC}")
    print(f"The remaining ones were deleted to save space.")