#controllers.py
import os
import sys
from collections import Counter
import requests
from plot_functions import *
from utils import *

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from gaussianPuff.Sensor import SensorSubstance, SensorAir
from gaussianPuff.config import NPS, OutputType, DispersionModelType, ModelConfig

API_URL = "http://127.0.0.1:"

def generate_binary_map(payload):
    response = requests.post(f"{API_URL}8001/generate_binary_map", json=payload)
    if response.status_code != 200:
        st.error("Error in binary map generation.")
        return None

    data = response.json()
    if data.get("status_code") != "success":
        st.error("Error in binary map generation.")
        return None

    binary_map = np.array(data.get("map"), dtype=np.float32)
    metadata = data.get("metadata", {})
    free_cells = np.argwhere(binary_map == 1)
    building_cells = np.sum(binary_map == 0)

    return binary_map, free_cells, building_cells, metadata

def meteorological_condition():

    sensor_air = SensorAir(sensor_id=00, x=0.0, y=0.0, z=2.0)
    wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = sensor_air.sample_meteorology()

    return wind_speed, wind_type, stability_type, stability_value, humidify, RH

def air_sample(n_sensors, free_cells, wind_speed, stability_type, RH, wind_type):
    sensors_substance = []

    for i in range(n_sensors):
        x, y = random_position(free_cells)
        sensor_substance = SensorSubstance(i, x=x, y=y, z=2.0,
                                           noise_level=round(np.random.uniform(0.0, 0.0005), 4))
        sensors_substance.append(sensor_substance)

        mass_spectrum = []
        for sensor in sensors_substance:
            recording = sensor.run_sensor(wind_speed, stability_type, RH, wind_type)
            recording = [rec for rec in recording if not np.isnan(rec).any()]
            mass_spectrum.extend(recording)

    return sensors_substance, mass_spectrum

def nps_classificator(mass_spectrum):
    if mass_spectrum:
        spectra_json = [m.tolist() for m in mass_spectrum]
        print(f"spectra_json: {type(spectra_json)}")
        response_dnn = requests.post(f"{API_URL}8000/predict_dnn", json={"spectra": spectra_json})

        if response_dnn.status_code == 200:
            predictions = response_dnn.json().get("predictions", [])
            print(len(predictions))
            substance_nps = [pred for pred in predictions if pred in nps_classes]
        else:
            st.error(f"Errore API {response_dnn.status_code}")

    print(type(substance_nps))
    print(len(substance_nps))

    if substance_nps:
        most_common_substance = Counter(substance_nps).most_common(1)[0][0]
        print(most_common_substance)
        nps = NPS.from_string(most_common_substance)
    else:
        most_common_substance = None
        print("Nessuna sostanza presente")

    return most_common_substance, nps

def gaussian_simulation(payload, free_cells, wind_speed, stability_type, RH,
                        wind_type, stability_value, dry_size, nps, humidify):

    x_src, y_src = random_position(free_cells)
    h_src = round(np.random.uniform(1, 10), 2)  # altezza del pennacchio
    Q = round(np.random.uniform(0.0001, 0.01), 4)  # tasso di emissione
    stacks = [(x_src, y_src, Q, h_src)]

    print(stability_value)
    print(wind_speed)
    print(wind_type)

    param_gaussian_model = ModelConfig(
        days=10,
        RH=RH,
        aerosol_type=NPS(nps),
        humidify=humidify,
        stability_profile=stability_type,
        stability_value=stability_value,
        wind_type=wind_type,
        wind_speed=wind_speed,
        output=OutputType.PLAN_VIEW,
        stacks=stacks,
        dry_size=dry_size, x_slice=26, y_slice=1,
        dispersion_model=DispersionModelType.PLUME)

    bounds = (payload["min_lon"], payload["min_lat"], payload["max_lon"], payload["max_lat"])

    response_gauss = requests.post(f"{API_URL}8002/start_simulation",
                                   json={"config": param_gaussian_model.to_dict(),
                                         "bounds": bounds})

    print("risposta ottenuta")
    print(f"code: {response_gauss.status_code}")
    print(response_gauss)

    if response_gauss.status_code != 200:
        return None

    gauss_data = response_gauss.json()
    x_raw = gauss_data.get("x", [])
    y_raw = gauss_data.get("y", [])
    times_raw = gauss_data.get("times", [])
    wind_dir_raw = gauss_data.get("wind_dir")
    C1_raw = gauss_data.get("concentration", [])

    x = np.array(x_raw)
    y = np.array(y_raw)
    times = np.array(times_raw)
    wind_dir = np.array(wind_dir_raw)
    C1 = np.array(C1_raw)

    return C1, x, y, times, wind_dir, C1_raw

def emission_source(sensors_substance, C1, x, y, times, wind_type, wind_dir, wind_speed):

    payload_sensors = []
    for s in sensors_substance:

        if not s.is_fault:

            s.sample_substance(C1, x, y, times)
            # s.sample_substance_synthetic()

            for idx, (t_idx, conc) in enumerate(zip(s.times, s.noisy_concentrations)):
                if idx >= len(wind_dir):
                    break
                wd = wind_dir[idx]

                payload_sensors.append({
                    "sensor_id": s.id,
                    "sensor_is_fault": s.is_fault,
                    "time": t_idx,
                    "conc": conc if not s.is_fault else None,
                    "wind_dir_x": np.cos(np.deg2rad(wd)) if not s.is_fault else None,
                    "wind_dir_y": np.sin(np.deg2rad(wd)) if not s.is_fault else None,
                    "wind_speed": wind_speed if not s.is_fault else None,
                    "wind_type": wind_type.value if not s.is_fault else None,
                })

    n_sensor_operating = ([s for s in sensors_substance if not s.is_fault]).__len__()

    response_loc = requests.post(f"{API_URL}8003/predict_source_raw", json={
        "payload_sensors": payload_sensors,
        "n_sensor_operating": n_sensor_operating
    })

    if response_loc.status_code != 200:
        st.error("Error in prediction of source.")

    data = response_loc.json()
    x = data["x"]
    y = data["y"]

    return  x, y

def correction_dispersion(payload):

    response_mcxm = requests.post(f"{API_URL}8001/correct_dispersion",
                                  json={
                                      "wind_speed": wind_speed,
                                      "wind_dir": wind_dir.tolist(),
                                      "concentration_map": C1.tolist(),
                                      "building_map": binary_map.tolist(),
                                      "global_features": list(metadata.values())
                                  })

    if response_mcxm.status_code != 200:
        st.error("Errore nella correzione della dispersione.")
        return sensors_substance, substance_nps, origin_lat, origin_lon, C1, metadata

    real_dispersion_map = response_mcxm.json().get("predictions", [])

    return real_dispersion_map