from config import ModelConfig, OutputType, WindType, StabilityType, NPS, PasquillGiffordStability, DispersionModelType, ConfigPuff
from gaussianModel import run_dispersion_model
import plot_utils
import os
from Sensor import SensorSubstance
import numpy as np
import json

BINARY_MAP_PATH = os.path.join("CorrectionDispersion/binary_maps_data/roma_italy_bbox.npy")
BINARY_MAP_PATH_METADATA= os.path.join("CorrectionDispersion/binary_maps_data/roma_italy_metadata_bbox.json")
N_SENSORS = 5

with open(BINARY_MAP_PATH_METADATA, 'r') as file:
    binary_map_metadata = json.load(file)

# Caricamento mappa binaria (1 = suolo libero, 0 = edificio)
binary_map = np.load(BINARY_MAP_PATH)
free_cells = np.argwhere(binary_map == 1)

def random_position():
    idx = np.random.choice(len(free_cells))
    y, x = free_cells[idx]
    return float(x), float(y)

if __name__ == "__main__":

    origin_lat_ref, origin_lon_ref = 41.894592, 12.488163  # lat/lon di riferimento della mappa

    origin_x, origin_y = random_position()  # coordinate in metri nella griglia

    origin_lat, origin_lon = plot_utils.meters_to_latlon(origin_x, origin_y, origin_lat_ref, origin_lon_ref)

    sensors = []
    for i in range(N_SENSORS):
        x, y=random_position()
        sensor = SensorSubstance(i, x=x ,y=y, z=2.0, noise_level=round(np.random.uniform(0.0, 0.0005), 4))
        sensors.append(sensor)

    x_slice=26
    y_slice=26

    config = ModelConfig(
        days=8,
        RH=0.01,
        aerosol_type=NPS.TRYPTAMINE_ANALOGUES,
        humidify=True,
        stability_profile=StabilityType.CONSTANT,
        stability_value=PasquillGiffordStability.MODERATELY_UNSTABLE,
        wind_type=WindType.FLUCTUATING,
        wind_speed=10.,
        output=OutputType.PLAN_VIEW,
        stacks=[(origin_lat, origin_lon, 0.0020, 4.68)],
        dry_size=1.0,
        x_slice=x_slice,
        y_slice=y_slice,
        #grid_size=binary_map_metadata['grid_size'][0],
        dispersion_model=DispersionModelType.PLUME,
        config_puff=ConfigPuff(puff_interval=1, max_puff_age=6)  # 1 hour interval, max age 6 hours
    )

    result = run_dispersion_model(config)#, bounds=binary_map_metadata['bounds'])
    C1, (x, y, z), times, stability, wind_dir, stab_label, wind_label, puff = result
    
    plot_utils.plot_plan_view(C1, x, y, f"Plan View - {stab_label} - {wind_label}", wind_dir, 10. ,puff, stability_class=PasquillGiffordStability.NEUTRAL.value)
    
    binary_map=binary_map[:,:, np.newaxis]  # aggiunta na dimensione per la compatibilità con C1
    
    mask_edifici=C1*binary_map
  
    plot_utils.plot_plan_view(mask_edifici, x, y, f"Plan View - {stab_label} - {wind_label}", wind_dir, 10. )
    plot_utils.plot_surface_time(C1, times, x_slice, y_slice, stability, stab_label, f"Surface Time - {stab_label} - {wind_label}")
    plot_utils.plot_height_slice(C1, y, z, stab_label, wind_label)

    for sensor in sensors:
        sensor.sample_substance(C1, x, y, times)
        sensor.plot_timeseries(use_noisy=True)
    
    sensor_data = [(sensor.x, sensor.y) for sensor in sensors]

    mappa = plot_utils.plot_concentration_with_sensors(C1, x, y, sensor_data, (origin_lat, origin_lon), times, title="Concentrazione con sensori")
    if mappa is not None:
        mappa.save("gaussianPuff/example/puff_map_with_sensors_0708.html")
        print("📍 Mappa salvata: puff_map_with_sensors_0708.html")
    else:
        print("⚠️ Errore: la funzione plot_concentration_with_sensors ha restituito None, impossibile salvare la mappa.")
    
    mappa=plot_utils.animate_plan_view(C1,x,y,binary_map, sensor_data, save_path="gaussianPuff\\example\\animated")
