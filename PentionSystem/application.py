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

API_CORRECTION = "http://correction_dispersion:8001"
API_CLASSIFICATORE = "http://clas_nps:8000"
API_GAUSSIAN = "http://gaussian_dispersion_model:8002"
API_SOURCE = "http://loc_emission_source:8003"

def safe_markdown(placeholder, text, level="info"):
    try:
        placeholder.markdown(text)
    except Exception as e:
        print(f"[WARNING] Placeholder update failed: {e}")
        if level == "error":
            st.error(text)
        elif level == "warning":
            st.warning(text)
        else:
            st.info(text)


def run_application(payload):
    
    n_sensors = payload.get("Number of sensors", 10)
    payload.pop("Number of sensors", None)

    progress = 0
    progress_bar.progress(progress)

    # --- Binary map generation
    status_text.text("Binary map generation...")

    response = requests.post(f"{API_CORRECTION}/generate_binary_map", json=payload)

    if response.status_code != 200:
        st.error("Error in binary map generation.")
        return None
    
    data = response.json()
    if data.get("status_code") != "success":
        st.error("Error in binary map generation.")
        return None
    
    binary_map = np.array(data.get("map"), dtype=np.float32)
    metadata = data.get("metadata", {})
    # dopo aver creato free_cells
    free_cells = np.argwhere(binary_map == 1)
    if free_cells.size == 0:
        st.error("La mappa binaria non contiene celle libere per posizionare sensori/sorgente.")
        return

    building_cells = np.sum(binary_map == 0)

    with metadata_section:
        metadata_placeholder.markdown(
            f"**Griglia**: {metadata.get('grid_size', 'N/A')}×{metadata.get('grid_size', 'N/A')}\n"
            f"**Edifici totali**: {metadata.get('total_buildings', 'N/A')}\n"
            f"**Celle edifici**: {int(np.sum(building_cells)) if isinstance(building_cells, np.ndarray) else building_cells:,}\n"
            f"**Celle libere**: {int(np.sum(free_cells)) if isinstance(free_cells, np.ndarray) else free_cells:,}\n"
            f"**CRS**: {metadata.get('crs', 'N/A')}\n"
            f"**Risoluzione**: {metadata.get('resolution (m)', 'N/A')} m\n"
            f"**Densità edifici**: {float(metadata.get('building_density', np.nan)):.1f}%\n"
            f"**Altezza media edifici**: {float(metadata.get('mean_height', np.nan))} m\n"
            f"**Città**: {metadata.get('city', 'N/A')}"
        )

    progress += 20
    progress_bar.progress(progress)

    # --- Meteo condition
    status_text.text("Sample meteo condition...")
    sensor_air = SensorAir(sensor_id=00, x=0.0, y=0.0, z=2.0)
    wind_speed, wind_type, stability_type, stability_value, humidify, dry_size, RH = sensor_air.sample_meteorology()

    if weather_section is not None:
        safe_markdown(weather_placeholder,
            f"💨 **Wind speed (m/s):** {wind_speed}  \n"
            f"💨 **Wind type:** {wind_type}  \n"
            f"📈 **Stability:** {stability_type}  \n"
            f"♒︎ **Relative Humidity (%):** {RH}"
        ) 

    # --- Sensor substance
    status_text.text("Air sampling...")
    sensors_substance = []
  
    for i in range(n_sensors):
        x, y = random_position(free_cells)
        sensor_substance = SensorSubstance(i, x=x, y=y, z=2.0,
                                           noise_level=round(np.random.uniform(0.0, 0.0005), 4))
        sensors_substance.append(sensor_substance)

    plot_binary_map(binary_map, metadata['bounds'], map_section, sensors_substance)

    mass_spectrum = []
    for sensor in sensors_substance:
        recording = sensor.run_sensor(wind_speed, stability_type, RH, wind_type)
        recording = [rec for rec in recording if not np.isnan(rec).any()]
        mass_spectrum.extend(recording)

    print(f"1->{type(mass_spectrum)}") # list
    print(f"2->{type(mass_spectrum[0])}") # numpy.ndarray
       
    if sensors_section is not None:
        sensor_info = [{"ID": s.id, "x": s.x, "y": s.y, "Status": "Operating" if not s.is_fault else "Faulty",}
                       for s in sensors_substance]
        sensors_placeholder.table(sensor_info)

    progress += 20
    progress_bar.progress(progress) 

    # --- NPS classification
    status_text.text("NPS classification...")
    substance_nps = []

    if mass_spectrum:
        spectra_json = [m.tolist() for m in mass_spectrum]
        print(f"spectra_json: {type(spectra_json)}")
        response_dnn = requests.post(f"{API_CLASSIFICATORE}/predict_dnn", json={"spectra": spectra_json})

        if response_dnn.status_code == 200:
            predictions = response_dnn.json().get("predictions", [])
            print(len(predictions))
            substance_nps = [pred for pred in predictions if pred in nps_classes]
        else:
            st.error(f"Errore API {response_dnn.status_code}")

    print(type(substance_nps))
    print(len(substance_nps))

    # subito dopo la classificazione
    if substance_nps:
        most_common_substance = Counter(substance_nps).most_common(1)[0][0]
        nps = NPS.from_string(most_common_substance)
    else:
        most_common_substance = None
        nps = NPS.OTHER_COMPOUNDS  # ✅ fallback sicuro per ModelConfig


    if nps_section is not None:
        if substance_nps:
            nps_placeholder.markdown(most_common_substance)
        else:
            nps_placeholder.warning("No NPS identified.")

    progress += 20
    progress_bar.progress(progress)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

    response_gauss = requests.post(f"{API_GAUSSIAN}/start_simulation",
                                   json={"config": param_gaussian_model.to_dict(),
                                         "bounds": bounds})

    print("risposta ottenuta")
    print(f"code: {response_gauss.status_code}")
    print(response_gauss)

    if response_gauss.status_code != 200:
        st.error("Error in Gaussian puff simulation 01.")
        return sensors_substance, substance_nps, None, None, None, metadata

    gauss_data = response_gauss.json()
    x_raw = gauss_data.get("x", [])
    y_raw = gauss_data.get("y", [])
    times_raw = gauss_data.get("times", [])
    wind_dir_raw = gauss_data.get("wind_dir")
    C1_raw = gauss_data.get("concentration", [])

    x=np.array(x_raw)
    y=np.array(y_raw)
    times=np.array(times_raw)
    wind_dir=np.array(wind_dir_raw)
    C1=np.array(C1_raw)

    print(type(C1))
    print(C1.shape)
    print(type(wind_dir))
    print(wind_dir.shape)
    print(type(wind_speed))
    print(type(x))
    print(x.shape)
    print(type(y))
    print(y.shape)

    status_text.text("Dispersion map generation...")
    plot_plan_view(C1, x, y, dispersion_placeholder)
    status_text.text("Wind rose graph generation...")
    plot_wind_rose(wind_dir, wind_speed, wind_rose_placeholder)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # --- Localizzazione sorgente
    status_text.text("Source estimation...")

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

    status_text.text("Start the prediction of the source...")
    response_loc = requests.post(f"{API_SOURCE}/predict_source_raw", json={
        "payload_sensors": payload_sensors,
        "n_sensor_operating": n_sensor_operating
    })

    if response_loc.status_code != 200:
        st.error("Error in prediction of source.")

    data = response_loc.json()
    x = data["x"]
    y = data["y"]

    if source_section is not None:
        if x is not None and y is not None:
            source_placeholder.markdown(f"Lat: {x}, Long: {y}")
        else:
            source_placeholder.warning("Source not estimated.")

    progress += 20
    progress_bar.progress(progress)

    # --- gaussian plume dispersion (raw simulation) 
    status_text.text("Raw dispersion simulation...")

    stacks = [(x, y, Q, h_src)]

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

    response_gauss = requests.post(f"{API_GAUSSIAN}/start_simulation",
                                   json={"config": param_gaussian_model.to_dict(),
                                         "bounds": bounds})
        
    if response_gauss.status_code != 200:
        st.error("Error in Gaussian puff simulation.")
        return sensors_substance, substance_nps, None, None, None, metadata

    gauss_data = response_gauss.json()
    x_raw = gauss_data.get("x", [])
    y_raw = gauss_data.get("y", [])
    times_raw = gauss_data.get("times", [])
    wind_dir_raw = gauss_data.get("wind_dir")
    C1_raw = gauss_data.get("concentration", [])

    x_grid = np.array(x_raw)
    y_grid = np.array(y_raw)
    times = np.array(times_raw)
    wind_dir = np.array(wind_dir_raw)
    C1 = np.array(C1_raw)

    status_text.text("Dispersion map generation...")
    plot_plan_view(C1, x_grid, y_grid, dispersion_placeholder)
    status_text.text("Wind rose graph generation...")
    plot_wind_rose(wind_dir, wind_speed, wind_rose_placeholder)

    progress += 20
    progress_bar.progress(progress)

    # --- Dispersion simulation + correction
    status_text.text("Dispersion simulation...")
    response_mcxm = requests.post(f"{API_CORRECTION}/correct_dispersion",
                                  json={
                                      "wind_speed": wind_speed,
                                      "wind_dir": wind_dir.tolist(),
                                      "concentration_map": C1.tolist(),
                                      "building_map": binary_map.tolist(),
                                      "global_features": None
                                  })

    if response_mcxm.status_code != 200: 
        st.error("Errore nella correzione della dispersione.") 
        return sensors_substance, substance_nps, x, y, C1, metadata
    
    real_dispersion_map = response_mcxm.json().get("predictions", [])
    real_dispersion_map = np.array(real_dispersion_map)
    print(f"mapp finale {type(real_dispersion_map)}")
    print(real_dispersion_map.shape)
    #progress += 20
    #progress_bar.progress(progress)

    from streamlit_folium import st_folium

    if map_section is not None:
        m = plot_dispersion_on_map(
            payload["min_lat"], payload["min_lon"],
            payload["max_lat"], payload["max_lon"],
            sensors_substance, real_dispersion_map, x, y
        )
        map_section.subheader("🗺️ Dispersion map")
        st_folium(m, width=700, height=500)
        m.save("dispersion_map.html")

    # plot_plan_view(real_dispersion_map, x_grid, y_grid, map_section)  # ← commentata

    progress = 100
    progress_bar.progress(progress)
    status_text.text("Simulation completed ✅")
    print("END")

    st.session_state.simulation_results = {
        "weather": {
            "wind_speed": wind_speed,
            "wind_type": wind_type,
            "stability": stability_type,
            "RH": RH,
            "wind_dir": wind_dir.tolist() if isinstance(wind_dir, np.ndarray) else wind_dir,  # ✅ NEW
        },
        "sensors": sensors_substance,
        "nps": most_common_substance,
        "source": (x, y),
        "dispersion_map": real_dispersion_map,
        "metadata": {
            **metadata,
            "bounds": (payload["min_lon"], payload["min_lat"], payload["max_lon"], payload["max_lat"]),  # ✅ NEW
        },
        "grid": {  # ✅ NEW: per ridisegnare la plan-view
            "x_grid": x_grid.tolist() if isinstance(x_grid, np.ndarray) else x_grid,
            "y_grid": y_grid.tolist() if isinstance(y_grid, np.ndarray) else y_grid,
        }
    }


    return {
        "weather": {
            "wind_speed": wind_speed,
            "wind_type": wind_type,
            "stability": stability_type,
            "RH": RH,
            "wind_dir": wind_dir.tolist() if isinstance(wind_dir, np.ndarray) else wind_dir,  # ✅ NEW
        },
        "sensors": sensors_substance,
        "nps": most_common_substance,
        "source": (x, y),
        "dispersion_map": real_dispersion_map,
        "metadata": {
            **metadata,
            "bounds": (payload["min_lon"], payload["min_lat"], payload["max_lon"], payload["max_lat"]),  # ✅ NEW
        },
        "grid": {  # ✅ NEW: per ridisegnare la plan-view
            "x_grid": x_grid.tolist() if isinstance(x_grid, np.ndarray) else x_grid,
            "y_grid": y_grid.tolist() if isinstance(y_grid, np.ndarray) else y_grid,
        }
    }


def render_results_from_state(results):
    # 1) Meteo
    if results.get("weather"):
        w = results["weather"]
        safe_markdown(
            weather_placeholder,
            f"- **Wind speed (m/s):** {w.get('wind_speed')}  \n"
            f"- **Wind type:** {w.get('wind_type')}  \n"
            f"- **Stability:** {w.get('stability')}  \n"
            f"- **Relative Humidity (%):** {w.get('RH')}"
        )
        # Wind rose
        if w.get("wind_dir") is not None and w.get("wind_speed") is not None:
            wd = np.array(w["wind_dir"])
            ws = w["wind_speed"]
            plot_wind_rose(wd, ws, wind_rose_placeholder)

    # 2) Sensori
    if results.get("sensors"):
        sensor_info = [{"ID": s.id, "x": s.x, "y": s.y,
                        "Status": "Operating" if not s.is_fault else "Faulty"}
                       for s in results["sensors"]]
        sensors_placeholder.table(sensor_info)

    # 3) NPS
    if results.get("nps") is not None:
        if results["nps"]:
            nps_placeholder.write(results["nps"])
        else:
            nps_placeholder.warning("No NPS identified.")

    # 4) Sorgente
    if results.get("source") is not None:
        origin_lat, origin_lon = results["source"]
        if origin_lat is not None and origin_lon is not None:
            source_placeholder.write(f"Lat: {origin_lat}, Long: {origin_lon}")
        else:
            source_placeholder.warning("Source not estimated.")

    # 5) Dispersione (plan view + folium)
    disp = results.get("dispersion_map")
    grid = results.get("grid", {})
    meta = results.get("metadata", {})
    bounds = meta.get("bounds") or (
        meta.get("min_lon"), meta.get("min_lat"), meta.get("max_lon"), meta.get("max_lat")
    )

    if disp is not None:
        # Plan view
        xg = np.array(grid.get("x_grid")) if grid.get("x_grid") is not None else None
        yg = np.array(grid.get("y_grid")) if grid.get("y_grid") is not None else None

        if xg is not None and yg is not None:
            plot_plan_view(np.array(disp), xg, yg, dispersion_placeholder)
        else:
            # fallback: griglie uniformi
            if bounds and all(v is not None for v in bounds) and isinstance(disp, np.ndarray):
                min_lon, min_lat, max_lon, max_lat = bounds
                ny, nx = disp.shape[:2]
                x_lin = np.linspace(min_lon, max_lon, nx)
                y_lin = np.linspace(min_lat, max_lat, ny)
                Xg, Yg = np.meshgrid(x_lin, y_lin)
                plot_plan_view(np.array(disp), Xg, Yg, dispersion_placeholder)

        # Folium map
        if bounds and all(v is not None for v in bounds):
            min_lon, min_lat, max_lon, max_lat = bounds
            m = plot_dispersion_on_map(
                min_lat, min_lon, max_lat, max_lon,
                results.get("sensors") or [], np.array(disp),
                *(results.get("source") or (None, None))
            )
            from streamlit_folium import st_folium
            st_folium(m, width=700, height=500)

    st.sidebar.success("✅ Simulation results loaded successfully")


# ---------------- INTERFACCIA STREAMLIT ---------------- #
st.set_page_config(page_title="PentionSystem", layout="wide")
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = {
        "weather": None,
        "sensors": None,
        "nps": None,
        "source": None,
        "dispersion_map": None,
        "metadata": None
    }

st.markdown(
    """
    <style>
        /* HEADER STICKY */
        .main-header {
            position: sticky;
            top: 0;
            background: linear-gradient(90deg, #3a0ca3, #4361ee, #4cc9f0);
            color: white;
            padding: 1.5rem 0;
            text-align: center;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 1px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.15);
            border-radius: 0 0 10px 10px;
            z-index: 999;
        }
    </style>
    <div class="main-header">
        💊 PENTION — NPS Emission Source Identification
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar input
st.sidebar.header("Insert simulation parameters")
min_lat = st.sidebar.number_input("Min Lat", value=41.89, format="%.5f")
min_lon = st.sidebar.number_input("Min Lon", value=12.48, format="%.5f")
max_lat = st.sidebar.number_input("Max Lat", value=41.91, format="%.5f")
max_lon = st.sidebar.number_input("Max Lon", value=12.50, format="%.5f")
place = st.sidebar.text_input("Place", value="Insert place name")
n_sensors = st.sidebar.slider("Number of sensors", min_value=5, max_value=50, value=10, step=1)

st.sidebar.markdown(
    """
    <style>
        /* Sidebar header */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 10%, #ffffff 90%);
            border-right: 2px solid #f1f3f5;
        }

        /* Start button */
        div[data-testid="stButton"] > button:first-child {
            background-color: #4cc9f0 !important;
            color: white !important;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            padding: 0.6em;
            transition: 0.3s;
        }
        div[data-testid="stButton"] > button:first-child:hover {
            background-color: #4895ef !important;
            transform: scale(1.03);
        }

        /* Stop button */
        div[data-testid="stButton"] > button:nth-child(2) {
            background-color: #ef233c !important;
            color: white !important;
            border-radius: 10px;
            border: none;
            font-weight: 600;
            padding: 0.6em;
            transition: 0.3s;
        }
        div[data-testid="stButton"] > button:nth-child(2):hover {
            background-color: #d90429 !important;
            transform: scale(1.03);
        }
    </style>
    """,
    unsafe_allow_html=True
)


col1, col2 = st.sidebar.columns(2)

with col1:
    st.markdown('<div class="start-btn">', unsafe_allow_html=True)
    start = st.button("▶ Start")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    stop = st.button("⏹ Stop")
    st.markdown('</div>', unsafe_allow_html=True)

# Layout colonne: lato-sinistra, centro (mappa), lato-destra
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌦 Meteo", "🧪 NPS Detected", "📍 Source", "🗺 Dispersion", "📡 Sensors"
])

with tab1:
    st.subheader("🌦 Meteo Conditions")
    weather_placeholder = st.empty()

with tab2:
    st.subheader("🧪 NPS Classification")
    nps_placeholder = st.empty()

with tab3:
    st.subheader("📍 Source Estimation")
    source_placeholder = st.empty()

with tab4:
    st.subheader("🌫 Dispersion Simulation & Wind Rose")
    dispersion_placeholder = st.empty()
    wind_rose_placeholder = st.empty()
    map_section = st.container()

with tab5:
    st.subheader("📡 Sensor Data")
    sensors_placeholder = st.empty()

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.caption("Developed for PENTION-S | Streamlit UI optimized")

# 🔧 Placeholder compatibili con la nuova UI
metadata_section = st.container()
metadata_placeholder = metadata_section.empty()

weather_section = tab1
nps_section = tab2
source_section = tab3
dispersion_section = tab4
sensors_section = tab5

# ---------------- START SIMULATION ---------------- #
if start:
    status_text.success("Simulation started ✅")

    payload = {
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
        "grid_size": 500,
        "place": place,
        "Number of sensors": n_sensors
    }

    with st.spinner("⏳ Running full simulation... please wait."):
        results = run_application(payload)
        st.session_state.simulation_results = results or st.session_state.simulation_results


elif stop:

    st.session_state.simulation_results = {
        "weather": None,
        "sensors": None,
        "nps": None,
        "source": None,
        "dispersion_map": None,
        "metadata": None
    }
    progress_bar.progress(0)
    status_text.text("Simulation stopped ❌")

    weather_placeholder.markdown(
        f"💨 **Wind speed (m/s):** N/A  \n"
        f"💨 **Wind type:** N/A  \n"
        f"📈 **Stability:** N/A  \n"
        f"♒︎ **Relative Humidity (%):** N/A"
    )
    sensors_placeholder.write("No data available.")
    nps_placeholder.write("N/A")
    source_placeholder.write("N/A")
    wind_rose_placeholder.empty()
    dispersion_placeholder.empty()
    map_section.empty()
else:
    results = st.session_state.simulation_results
    if results and any(v is not None for v in results.values()):
        render_results_from_state(results)
