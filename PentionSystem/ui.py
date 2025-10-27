import streamlit as st
from controllers import *

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
    <div style="
        position: sticky; 
        top: 0; 
        background-color: white; 
        padding: 20px; 
        z-index: 999; 
        font-size: 36px; 
        font-weight: bold;
        text-align: center;
    ">
        💊 PENTION - NPS Source emission identification system
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
    .start-btn > button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
        padding: 0.5em 0;
    }
    .stop-btn > button {
        background-color: #dc3545 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
        padding: 0.5em 0;
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
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_left:
    weather_section = st.container()
    dispersion_section = st.container()
    metadata_section = st.container()

with col_center:
    map_section = st.container()

with col_right:
    nps_section = st.container()
    source_section = st.container()
    wind_rose_section = st.container()

with weather_section:
    st.markdown("**⛅Meteo conditions**")
    weather_placeholder = st.empty()
    weather_placeholder.markdown(
        f"💨 **Wind speed (m/s):** N/A  \n"
        f"💨 **Wind type:** N/A  \n"
        f"📈 **Stability:** N/A  \n"
        f"♒︎ **Relative Humidity (%):** N/A"
    )

with dispersion_section:
    st.markdown("**🗺️ Dispersion map**")
    dispersion_placeholder = st.empty()

sensors_section = st.container()

with metadata_section:
    st.markdown("**🏙️ Info city map**")
    metadata_placeholder = st.empty()

with sensors_section:
    st.markdown("**🛰️ Sensor**")
    sensors_placeholder = st.empty()
    sensors_placeholder.write("No data available.")

with nps_section:
    st.markdown("**🧪 Nps predicted by sensor**")
    nps_placeholder = st.empty()
    nps_placeholder.write("N/A")

with source_section:
    st.markdown("**📍 Source estimated**")
    source_placeholder = st.empty()
    source_placeholder.write("N/A")

with wind_rose_section:
    st.markdown("🧭 **Wind rose**")
    wind_rose_placeholder = st.empty()

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

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

    run_application(payload)

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

    if results["weather"] is not None:
        weather_placeholder.markdown(
            f"- **Wind speed (m/s):** {results['weather']['wind_speed']}  \n"
            f"- **Wind type:** {results['weather']['wind_type']}  \n"
            f"- **Stability:** {results['weather']['stability']}  \n"
            f"- **Relative Humidity (%):** {results['weather']['RH']}"
        )

    if results["sensors"] is not None:
        sensor_info = [{"ID": s.id, "x": s.x, "y": s.y, "Status": "Operating" if not s.is_fault else "Faulty"}
                       for s in results["sensors"]]
        sensors_placeholder.table(sensor_info)

    if results["nps"] is not None:
        if results["nps"]:
            nps_placeholder.write(results["nps"])
        else:
            nps_placeholder.warning("No NPS identified.")

    if results["source"] is not None:
        origin_lat, origin_lon = results["source"]
        if origin_lat is not None and origin_lon is not None:
            source_placeholder.write(f"Lat: {origin_lat}, Long: {origin_lon}")
        else:
            source_placeholder.warning("Source not estimated.")