import datetime
import numpy as np
import pandas as pd
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gaussianPuff')))
from config import ModelConfig, StabilityType, WindType, OutputType, NPS, PasquillGiffordStability, DispersionModelType, ConfigPuff
from gaussianModel import run_dispersion_model
from Sensor import SensorSubstance, SensorAir

# Parametri generali
N_SIMULATIONS = 100
N_SENSORS = 10
SAVE_DIR = "./CorrectionDispersion/dataset"
SAVE_DIR_CONC_REAL= f"./CorrectionDispersion/dataset/real_dispersion_2025_09_10"
BINARY_MAP_PATH = os.path.join(os.path.dirname(__file__), "binary_maps_data/roma_italy_bbox.npy")

os.makedirs(SAVE_DIR, exist_ok=True)

# Caricamento mappa binaria (1 = suolo libero, 0 = edificio)
binary_map = np.load(BINARY_MAP_PATH)
free_cells = np.argwhere(binary_map == 1)

def random_position():
    idx = np.random.choice(len(free_cells))
    y, x = free_cells[idx]
    return float(x), float(y)

def assign_wind_speed(stability: PasquillGiffordStability) -> float:
    """
    Restituisce una velocità del vento (m/s) coerente con la stabilità atmosferica.
    I range sono basati su letteratura meteorologica semplificata.
    """
    if stability == PasquillGiffordStability.VERY_UNSTABLE:  # A
        return round(random.uniform(2.0, 6.0), 2)
    elif stability == PasquillGiffordStability.MODERATELY_UNSTABLE:  # B
        return round(random.uniform(2.0, 5.0), 2)
    elif stability == PasquillGiffordStability.SLIGHTLY_UNSTABLE:  # C
        return round(random.uniform(3.0, 6.5), 2)
    elif stability == PasquillGiffordStability.NEUTRAL:  # D
        return round(random.uniform(4.0, 8.0), 2)
    elif stability == PasquillGiffordStability.MODERATELY_STABLE:  # E
        return round(random.uniform(1.0, 4.0), 2)
    elif stability == PasquillGiffordStability.VERY_STABLE:  # F
        return round(random.uniform(0.5, 3.0), 2)
    else:
        return round(random.uniform(2.0, 6.0), 2)

def sample_meteorology():
    wind_type = random.choice([WindType.CONSTANT ,WindType.PREVAILING, WindType.FLUCTUATING])
    stability_type = StabilityType.CONSTANT
    stability_value =  random.choice([PasquillGiffordStability.VERY_UNSTABLE, 
                                      PasquillGiffordStability.MODERATELY_UNSTABLE, 
                                      PasquillGiffordStability.SLIGHTLY_UNSTABLE, 
                                      PasquillGiffordStability.NEUTRAL,  
                                      PasquillGiffordStability.MODERATELY_STABLE, 
                                      PasquillGiffordStability.VERY_STABLE]) if stability_type == StabilityType.CONSTANT else 0
    
    wind_speed = assign_wind_speed(stability_value) # type: ignore

    return wind_speed, wind_type, stability_type, stability_value

def fault_probability(wind_speed, stability_value, RH, wind_type):
    # Base probability
    base_prob = 0.1
    
    # Aumenta probabilità con vento forte
    if wind_speed > 6.0:
        base_prob += 0.2
    
    # Aumenta probabilità se stabilità molto instabile o molto stabile
    if stability_value in [PasquillGiffordStability.VERY_UNSTABLE, PasquillGiffordStability.VERY_STABLE]:
        base_prob += 0.15
        
    # Aumenta probabilità con alta umidità
    if RH > 0.8:
        base_prob += 0.2
        
    # Vento fluttuante aumenta la probabilità
    if wind_type == WindType.FLUCTUATING:
        base_prob += 0.1
        
    # Limita la probabilità al massimo di 0.75 per evitare valori troppo estremi
    return min(base_prob, 0.75)

data_records = []
tsf_records = []

for i in range(N_SIMULATIONS):
    print(f"Simulazione {i+1}/{N_SIMULATIONS}")

    # Sorgente (stack): posizione, altezza, emissione
    x_src, y_src = random_position()
    h_src = round(np.random.uniform(1, 10), 2)  # altezza del pennacchio 
    Q = round(np.random.uniform(0.0001, 0.01), 4)  # tasso di emissione 
    stacks = [(x_src, y_src, Q, h_src)]

    sensor_air = SensorAir(sensor_id=0, x=0.0, y=0.0, z=2.0)

    # Meteo
    wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = sensor_air.sample_meteorology()

    # Sensori
    sensors = []
    for j in range(N_SENSORS):
        x, y=random_position()

        fault_prob = fault_probability(wind_speed, stability_value, 0.5, wind_type) 
        is_fault = random.random() < fault_prob

        sensor = SensorSubstance(
            j,
            x=x,
            y=y,
            z=random.uniform(1.5, 3.0),
            noise_level=round(np.random.uniform(0.0, 0.0005), 4)
        )
        sensors.append(sensor)

    # nps considerato casuale
    aerosol_type = random.choice(list(NPS))

    # humidify
    #humidify = random.choice([True, False])

    days=10
    config = ModelConfig(
        days=days,
        aerosol_type=aerosol_type,
        dry_size=1.0,
        humidify=humidify,
        RH=round(np.random.uniform(0, 0.99),2) if humidify else 0.0,
        stability_profile=stability_type,
        stability_value=stability_value, # type: ignore
        wind_type=wind_type,
        wind_speed=wind_speed,
        output=OutputType.PLAN_VIEW,
        stacks=stacks,
        x_slice=26,
        y_slice=1,
        dispersion_model=DispersionModelType.PLUME,
    )

    # Calcola concentrazioni con modello gaussiano
    C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = run_dispersion_model(config)
 
    filename = f"sim_{i}_conc_real_{datetime.datetime.now().date()}.npy"
    np.save(os.path.join(SAVE_DIR_CONC_REAL, filename), C1)

    # Salva record per ogni sensore e ogni simulazione
    for sensor in sensors:

        sensor.sample_substance(C1,x,y,times) 

        row = {
            "simulation_id": i,
            "sensor_id": sensor.id,
            "sensor_x": sensor.x,
            "sensor_y": sensor.y,
            "sensor_noise": sensor.noise_level,
            "sensor_height": sensor.z,
            "sensor_is_fault": sensor.is_fault,
            "RH": config.RH,
            "humidify": config.humidify,
            "days_simulation": days,
            "wind_type": wind_type.name,
            "wind_speed": wind_speed,
            "wind_dir": ",".join(map(str, wind_dir.tolist())),
            "stability_profile": stability_type.name,
            "stability_value": stability_value.name, # type: ignore
            "aerosol_type": aerosol_type.name,
            "source_x": x_src,
            "source_y": y_src,
            "source_h": h_src,
            "emission_rate": Q,
            "real_concentration_name_file": filename,
        }
        data_records.append(row)

        if len(sensor.noisy_concentrations) == 0:
            tsf_records.append({
                "simulation_id": i,
                "sensor_id": sensor.id,
                "sensor_is_fault": sensor.is_fault,
                "time": np.nan,
                "conc": np.nan,
                "wind_dir_x": np.nan,
                "wind_dir_y": np.nan,
                "wind_speed": np.nan,
                "wind_type": np.nan,
            })
        else:
            for idx, (t_idx, conc) in enumerate(zip(sensor.times, sensor.noisy_concentrations)):
                
                if idx >= len(wind_dir):
                    break  
                wd = wind_dir[idx]

                tsf_records.append({
                    "simulation_id": i,
                    "sensor_id": sensor.id,
                    "sensor_is_fault": sensor.is_fault,
                    "time": t_idx, 
                    "conc": conc if not sensor.is_fault else np.nan,
                    "wind_dir_x": np.cos(np.deg2rad(wd)) if not sensor.is_fault else np.nan, 
                    "wind_dir_y": np.sin(np.deg2rad(wd)) if not sensor.is_fault else np.nan, 
                    "wind_speed": wind_speed if not sensor.is_fault else np.nan,
                    "wind_type": wind_type.value if not sensor.is_fault else np.nan,
                })

# Salvataggio CSV
df = pd.DataFrame(data_records)
csv_path = os.path.join(SAVE_DIR, f"nps_simulated_dataset_gaussiano_{datetime.datetime.now().date()}.csv")
df.to_csv(csv_path, index=False)

df_tsf = pd.DataFrame(tsf_records)
tsf_csv_path = os.path.join(SAVE_DIR, f"nps_simulated_dataset_tsfresh_{datetime.datetime.now().date()}.csv")
df_tsf.to_csv(tsf_csv_path, index=False)

print(f"\nDataset generato e salvato in {csv_path} e {tsf_csv_path}")
