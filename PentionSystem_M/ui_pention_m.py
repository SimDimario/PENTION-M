# ui_pention_m.py
import asyncio
import json
import os
import random
from datetime import datetime
from glob import glob
from math import radians, sin, cos, sqrt, atan2

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

import networkx as nx
import osmnx as ox

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
AMSTERDAM_PLACE = "Amsterdam, Netherlands"
CACHE_DIR = "/app/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

GRAPH_PATH = os.path.join(CACHE_DIR, "amsterdam_drive.graphml")
DETECTION_RADIUS_M = 250.0  # raggio operativo per "detection"
STEP_DELAY_SEC = 0.5        # tempo tra un passo e l'altro
LOG_DIR = "/logs"
INGESTION_URL = "http://mlops_ingestion:8011/ingest_data"

app = FastAPI(title="PENTION-M UI (Van Simulation)")

# ----------------------------------------------------
# STATO GLOBALE SEMPLICE
# ----------------------------------------------------
class SimulationState:
    def __init__(self):
        self.G: nx.MultiDiGraph | None = None
        self.source_node = None
        self.van_node = None
        self.running = False
        self.detected = False
        self.path = []  # lista di (lat, lon)
        self.current_sim_id = None

state = SimulationState()
active_sockets: list[WebSocket] = []

# ----------------------------------------------------
# FUNZIONI DI SUPPORTO
# ----------------------------------------------------
def load_graph():
    """Carica il grafo stradale di Amsterdam o, in fallback, un grafo sintetico a griglia."""
    if state.G is not None:
        return state.G

    G = None

    # 1) Prova a caricare da cache
    if os.path.exists(GRAPH_PATH):
        try:
            print(f"[UI] Loading graph from cache: {GRAPH_PATH}", flush=True)
            G = ox.load_graphml(GRAPH_PATH)
            G = nx.Graph(G)
        except Exception as e:
            print(f"[UI] Error loading cached graph: {e}", flush=True)
            G = None

    # 2) Se non ho il grafo, provo a scaricarlo (se c'è internet nel container)
    if G is None:
        try:
            print("[UI] Downloading graph from OpenStreetMap...", flush=True)
            G = ox.graph_from_place(AMSTERDAM_PLACE, network_type="drive")
            ox.save_graphml(G, GRAPH_PATH)
            G = nx.Graph(G)
            print("[UI] Graph downloaded and cached.", flush=True)
        except Exception as e:
            print(f"[UI] ERROR downloading graph, falling back to synthetic grid: {e}", flush=True)
            G = None

    # 3) Fallback definitivo: grafo sintetico a griglia (nessun accesso OSM necessario)
    if G is None:
        print("[UI] Building synthetic grid graph for Amsterdam area.", flush=True)
        G = nx.grid_2d_graph(20, 20)  # 20x20 = 400 nodi

        # bounding box approssimativa su Amsterdam
        lat_min, lat_max = 52.35, 52.39
        lon_min, lon_max = 4.88, 4.92

        for (i, j) in G.nodes:
            fi = i / 19.0
            fj = j / 19.0
            lat = lat_min + (lat_max - lat_min) * fi
            lon = lon_min + (lon_max - lon_min) * fj
            G.nodes[(i, j)]["y"] = float(lat)
            G.nodes[(i, j)]["x"] = float(lon)

    state.G = G
    return G

def node_latlon(G, node):
    data = G.nodes[node]
    return float(data["y"]), float(data["x"])  # (lat, lon)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0  # raggio terrestre in m
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

async def broadcast(message: dict):
    """Invia un messaggio JSON a tutti i client WebSocket connessi."""
    dead = []
    for ws in active_sockets:
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in active_sockets:
            active_sockets.remove(ws)

def get_last_monitoring():
    """Legge l'ultima entry da monitoring_log.jsonl, se esiste."""
    log_path = os.path.join(LOG_DIR, "monitoring_log.jsonl")
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception:
        return None

def get_model_registry():
    """Legge model_registry.json, se esiste."""
    reg_path = os.path.join(LOG_DIR, "model_registry.json")
    if not os.path.exists(reg_path):
        return None
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def get_last_forensic_bundle():
    """Prende l'ultimo bundle forensic JSON in /logs/forensic."""
    dir_forensic = os.path.join(LOG_DIR, "forensic")
    if not os.path.isdir(dir_forensic):
        return None
    files = sorted(
        glob(os.path.join(dir_forensic, "bundle_*.json")),
        key=os.path.getmtime
    )
    if not files:
        return None
    try:
        with open(files[-1], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def build_simulation_payload(sim_id: str, lat: float, lon: float):
    """
    Costruisce un payload compatibile con SimulationData di api_ingestion.py
    (sample_simulation_data.json come riferimento).
    """
    now_iso = datetime.utcnow().isoformat() + "Z"

    # parametri "sensati ma fittizi" – il punto è attivare la pipeline reale
    temperature = round(random.uniform(15.0, 25.0), 1)
    humidity = round(random.uniform(0.4, 0.8), 2)
    wind_speed = round(random.uniform(2.0, 7.0), 1)
    wind_dir_deg = random.choice([90, 135, 180, 225, 270])
    stability_class = random.choice(["B", "C", "D", "NEUTRAL"])

    compound_name = random.choice(["Cathinone", "Cannabinoid", "Fentanyl analogue"])
    conc_series = [round(x, 4) for x in [0.002, 0.004, 0.0035, 0.0021]]

    sigma_y = round(random.uniform(0.1, 0.3), 3)
    sigma_z = round(random.uniform(0.05, 0.2), 3)
    pe_number = round(random.uniform(0.8, 1.5), 2)
    stability_idx = round(random.uniform(3.0, 5.0), 2)

    dispersion_map_id = f"map_{sim_id}"

    payload = {
        "simulation_id": sim_id,
        "timestamp": now_iso,
        "SensorAir": {
            "temperature_C": temperature,
            "humidity_%": humidity,  # alias corretto per humidity_
            "wind_speed_mps": wind_speed,
            "wind_dir_deg": wind_dir_deg,
            "stability_class": stability_class,
        },
        "SensorSubstance": {
            "compound_name": compound_name,
            "molecular_formula": "C9H11NO",
            "concentration_series_mg_m3": conc_series,
            "unit": "mg/m3",
            "noise_level": 0.05,
        },
        "SensorGPS": {
            "latitude": lat,
            "longitude": lon,
            "altitude_m": 2.0,
        },
        "PIML_Features": {
            "sigma_y": sigma_y,
            "sigma_z": sigma_z,
            "pe_number": pe_number,
            "wind_vector": [wind_speed, wind_dir_deg],
            "stability_index": stability_idx,
        },
        "Inference": {
            "dispersion_map_id": dispersion_map_id,
            "predicted_source_location": [0.0, 0.0],
            "predicted_class": compound_name,
            "confidence_score": 0.93,
        },
        "Monitoring": {
            "model_version": "PIML_v1.0",
            "drift_score": 0.0,
            "latency_ms": 0,
            "mse_free": 0.0,
        },
        "ModelOps": {
            "model_registry_id": "mdl_pention_m_ui",
            "training_data_version": "PIML_DS_v1",
            "retraining_trigger": False,
        },
        "UI_Output": {
            "dashboard_tabs": [
                "Simulation",
                "Dispersion",
                "Source",
                "NPS",
                "MLOps Monitoring",
            ],
            "visualization_files": ["dispersion_map.html", "wind_rose.png"],
        },
        "ForensicExport": {
            "export_file": f"{sim_id}_bundle.zip",
            "hash_sha256": "dummy_hash",
            "signature": "dummy_signature",
            "compliance_tags": ["GDPR", "LEA_audit_ok"],
        },
    }
    return payload

def call_ingestion_pipeline(sim_id: str, lat: float, lon: float):
    """
    Chiamata sincrona a /ingest_data. Il van si ferma finché non finisce.
    """
    payload = build_simulation_payload(sim_id, lat, lon)
    try:
        resp = requests.post(INGESTION_URL, json=payload, timeout=180)
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return {"code": resp.status_code, "body": body}
    except Exception as e:
        return {"code": 500, "body": {"error": str(e)}}

# ----------------------------------------------------
# LOOP DI SIMULAZIONE VAN
# ----------------------------------------------------

async def simulation_loop(force_near=False):
    try:
        G = load_graph()

        state.running = True
        state.detected = False
        state.path = []

        # ------------------------------
        # SCEGLIAMO NODO SORGENTE
        # ------------------------------
        nodes = list(G.nodes)
        if not nodes:
            raise RuntimeError("Graph has no nodes – cannot start simulation.")

        state.source_node = random.choice(nodes)
        source_lat, source_lon = node_latlon(G, state.source_node)

        # ------------------------------
        # SCEGLIAMO POSIZIONE VAN
        # ------------------------------
        if force_near:
            # scegli un nodo a distanza 600–800 m dalla sorgente
            candidates = []
            for n in nodes:
                lat, lon = node_latlon(G, n)
                dist = haversine_m(lat, lon, source_lat, source_lon)
                if 600 < dist < 800:
                    candidates.append(n)

            if candidates:
                state.van_node = random.choice(candidates)
            else:
                # fallback: vicino alla sorgente
                neighbors = list(G.neighbors(state.source_node))
                if neighbors:
                    state.van_node = random.choice(neighbors)
                else:
                    state.van_node = random.choice(nodes)
        else:
            # caso normale: van in un punto random lontano
            while True:
                candidate = random.choice(nodes)
                if candidate != state.source_node:
                    state.van_node = candidate
                    break

        van_lat, van_lon = node_latlon(G, state.van_node)
        state.path.append((van_lat, van_lon))

        # ------------------------------
        # ID SIMULAZIONE
        # ------------------------------
        sim_id = f"SIM_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        state.current_sim_id = sim_id

        # ------------------------------
        # INVIA INIT ALLA UI
        # ------------------------------
        await broadcast({
            "type": "init",
            "simulation_id": sim_id,
            "source": {"lat": source_lat, "lon": source_lon},
            "van": {"lat": van_lat, "lon": van_lon},
            "status": "patrolling",
        })

        # ==========================================================
        # == DEBUG MODE → PERCORSO DIRETTO (DIJKSTRA) VERSO LA SOURCE
        # ==========================================================
        if force_near:
            try:
                path_nodes = nx.shortest_path(
                    G,
                    source=state.van_node,
                    target=state.source_node,
                    weight="length"
                )
            except Exception:
                path_nodes = [state.van_node, state.source_node]

            # seguiamo il percorso
            for pn in path_nodes[1:]:
                if not state.running:
                    break

                state.van_node = pn
                lat, lon = node_latlon(G, pn)
                state.path.append((lat, lon))

                dist = haversine_m(lat, lon, source_lat, source_lon)

                await broadcast({
                    "type": "van_update",
                    "lat": lat,
                    "lon": lon,
                    "status": "patrolling",
                    "distance_m": dist
                })

                if dist <= DETECTION_RADIUS_M:
                    state.detected = True
                    state.running = False
                    break

                await asyncio.sleep(STEP_DELAY_SEC)

            # detection? → esegui pipeline
            if state.detected:
                result = call_ingestion_pipeline(sim_id, lat, lon)
                monitoring = get_last_monitoring()
                registry = get_model_registry()
                bundle = get_last_forensic_bundle()

                await broadcast({
                    "type": "detection_result",
                    "simulation_id": sim_id,
                    "ingestion_response": result,
                    "monitoring": monitoring,
                    "registry": registry,
                    "forensic_bundle": bundle,
                })

            return  # 🔚 termina la simulazione DEBUG

        # ==========================================================
        # == SIMULAZIONE NORMALE (RANDOM WALK)
        # ==========================================================
        while state.running and not state.detected:
            neighbors = list(G.neighbors(state.van_node))
            if not neighbors:
                break

            state.van_node = random.choice(neighbors)
            van_lat, van_lon = node_latlon(G, state.van_node)
            state.path.append((van_lat, van_lon))

            dist = haversine_m(van_lat, van_lon, source_lat, source_lon)

            if dist <= DETECTION_RADIUS_M:
                state.detected = True
                state.running = False

                await broadcast({
                    "type": "van_update",
                    "lat": van_lat,
                    "lon": van_lon,
                    "status": "detected",
                    "distance_m": dist,
                })

                result = call_ingestion_pipeline(sim_id, van_lat, van_lon)
                monitoring = get_last_monitoring()
                registry = get_model_registry()
                bundle = get_last_forensic_bundle()

                await broadcast({
                    "type": "detection_result",
                    "simulation_id": sim_id,
                    "ingestion_response": result,
                    "monitoring": monitoring,
                    "registry": registry,
                    "forensic_bundle": bundle,
                })
                break

            await broadcast({
                "type": "van_update",
                "lat": van_lat,
                "lon": van_lon,
                "status": "patrolling",
                "distance_m": dist,
            })

            await asyncio.sleep(STEP_DELAY_SEC)

        # fine corsa senza detection
        if not state.detected:
            await broadcast({
                "type": "sim_end",
                "simulation_id": sim_id,
                "reason": "stopped_or_completed",
            })

    except Exception as e:
        # se succede QUALSIASI errore, fermiamo la simulazione e avvisiamo la UI
        state.running = False
        state.detected = False
        print(f"[UI] simulation_loop error: {e}", flush=True)
        await broadcast({
            "type": "error",
            "message": str(e),
        })
# ----------------------------------------------------
# ENDPOINT HTTP
# ----------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>PENTION-M – Van Simulation (Amsterdam)</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    html, body {
      margin: 0;
      padding: 0;
      height: 100%;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0b1020;
      color: #f5f5f5;
    }
    #app {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 0;
      height: 100vh;
    }
    #map {
      height: 100%;
      width: 100%;
    }
    #sidebar {
      background: #111628;
      padding: 16px;
      box-sizing: border-box;
      border-left: 1px solid #22293b;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    h1 {
      font-size: 18px;
      margin: 0 0 4px 0;
    }
    .small {
      font-size: 12px;
      color: #9ba3b4;
    }
    button {
      padding: 8px 12px;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      background: #2563eb;
      color: white;
      font-weight: 500;
      margin-right: 8px;
    }
    button.secondary {
      background: #4b5563;
    }
    button:disabled {
      opacity: 0.5;
      cursor: default;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      background: #1f2937;
      margin-top: 4px;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      margin-right: 6px;
    }
    .status-patrolling .status-dot { background: #10b981; }
    .status-detected .status-dot { background: #f97316; }
    .status-idle .status-dot { background: #6b7280; }

    .card {
      background: #111827;
      border-radius: 12px;
      padding: 10px 12px;
      border: 1px solid #1f2937;
      font-size: 13px;
    }
    .card h2 {
      margin: 0 0 6px 0;
      font-size: 14px;
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 2px;
    }
    .metric-label {
      color: #9ca3af;
    }
    .metric-value {
      font-weight: 500;
    }
    .log-box {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 11px;
      max-height: 180px;
      overflow: auto;
      background: #020617;
      padding: 8px;
      border-radius: 8px;
      border: 1px solid #1f2937;
      white-space: pre-wrap;
    }
    @media (max-width: 900px) {
      #app {
        grid-template-columns: 1fr;
        grid-template-rows: 1.3fr 1fr;
      }
      #sidebar {
        border-left: none;
        border-top: 1px solid #22293b;
      }
    }
    #loading-screen {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: #0b1020ee;
      display: flex;
      justify-content: center;
      align-items: center;
      color: #ffffff;
      font-size: 20px;
      z-index: 9999;
      visibility: hidden;
    }
    .spinner {
      border: 4px solid #1e293b;
      border-top: 4px solid #38bdf8;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
<div id="loading-screen">
  <div class="spinner"></div>
  <div style="margin-left: 10px;">Loading map…</div>
</div>
<div id="app">
  <div id="map"></div>
  <div id="sidebar">      
    <div>
      <h1>PENTION-M – Van Simulation</h1>
      <div class="small">Amsterdam • mobile lab • PIML + MLOps pipeline</div>
      <div id="status-pill" class="status-pill status-idle">
        <div class="status-dot"></div>
        <span id="status-text">Idle</span>
      </div>
    </div>

    <div>
      <button id="btn-start">Start simulation</button>
      <button id="btn-reset" class="secondary" disabled>Reset</button>
      <button id="btn-debug" class="secondary">Debug: Start near source</button>
    </div>

    <div class="card">
      <h2>Simulation info</h2>
      <div class="metric-row">
        <div class="metric-label">Simulation ID</div>
        <div class="metric-value" id="sim-id">–</div>
      </div>
      <div class="metric-row">
        <div class="metric-label">Distance to source</div>
        <div class="metric-value" id="sim-distance">–</div>
      </div>
      <div class="metric-row">
        <div class="metric-label">Model version</div>
        <div class="metric-value" id="model-version">–</div>
      </div>
      <div class="metric-row">
        <div class="metric-label">Drift score</div>
        <div class="metric-value" id="drift-score">–</div>
      </div>
      <div class="metric-row">
        <div class="metric-label">Latency</div>
        <div class="metric-value" id="latency-ms">–</div>
      </div>
    </div>

    <div class="card">
      <h2>Last forensic bundle</h2>
      <div class="small" id="bundle-name">–</div>
      <div class="log-box" id="bundle-log">No bundle yet.</div>
    </div>

    <div class="card">
      <h2>Raw monitoring log (latest)</h2>
      <div class="log-box" id="monitoring-log">No monitoring event yet.</div>
    </div>
  </div>
</div>

<script
  src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
  integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
  crossorigin=""
></script>
<script>
  let map;
  let vanMarker = null;
  let sourceCircle = null;
  let vanPath = null;
  let pathLatLngs = [];
  let ws = null;

  const btnStart = document.getElementById("btn-start");
  const btnReset = document.getElementById("btn-reset");
  const statusPill = document.getElementById("status-pill");
  const statusText = document.getElementById("status-text");
  const simIdEl = document.getElementById("sim-id");
  const simDistEl = document.getElementById("sim-distance");
  const modelVersionEl = document.getElementById("model-version");
  const driftScoreEl = document.getElementById("drift-score");
  const latencyEl = document.getElementById("latency-ms");
  const bundleNameEl = document.getElementById("bundle-name");
  const bundleLogEl = document.getElementById("bundle-log");
  const monitoringLogEl = document.getElementById("monitoring-log");

  document.getElementById("btn-debug").addEventListener("click", async () => {
    showLoading();
    btnStart.disabled = true;
    btnReset.disabled = true;

    try {
      const resp = await fetch("/api/start_simulation_near", { method: "POST" });
      if (!resp.ok) {
        hideLoading();
        btnStart.disabled = false;
        btnReset.disabled = false;
        alert("Error starting debug simulation: " + resp.status);
      }
    } catch (e) {
      hideLoading();
      btnStart.disabled = false;
      btnReset.disabled = false;
      alert("Error starting debug simulation: " + e);
    }
  });

  function showLoading() {
    document.getElementById("loading-screen").style.visibility = "visible";
  }

  function hideLoading() {
    document.getElementById("loading-screen").style.visibility = "hidden";
  }

  function setStatus(state) {
    statusPill.classList.remove("status-idle", "status-patrolling", "status-detected");
    if (state === "patrolling") {
      statusPill.classList.add("status-patrolling");
      statusText.textContent = "Patrolling (van in movimento)";
    } else if (state === "detected") {
      statusPill.classList.add("status-detected");
      statusText.textContent = "Detection – pipeline in esecuzione";
    } else {
      statusPill.classList.add("status-idle");
      statusText.textContent = "Idle";
    }
  }

  function ensureMap() {
    if (!map) {
      map = L.map("map").setView([52.372, 4.900], 13);

      const tileLayer = L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
          maxZoom: 19,
          attribution: "&copy; OpenStreetMap contributors",
        }
      );

      tileLayer.addTo(map);

      hideLoading();
    }
  }

  function resetGraphics() {
    if (vanMarker) {
      map.removeLayer(vanMarker);
      vanMarker = null;
    }
    if (sourceCircle) {
      map.removeLayer(sourceCircle);
      sourceCircle = null;
    }
    if (vanPath) {
      map.removeLayer(vanPath);
      vanPath = null;
    }
    pathLatLngs = [];
    simIdEl.textContent = "–";
    simDistEl.textContent = "–";
  }

  function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      return;
    }

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${window.location.host}/ws`);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === "init") {
        simIdEl.textContent = msg.simulation_id || "–";
        setStatus("patrolling");
        btnStart.disabled = true;
        btnReset.disabled = false;

        resetGraphics();
        const sLat = msg.source.lat;
        const sLon = msg.source.lon;
        const vLat = msg.van.lat;
        const vLon = msg.van.lon;

        sourceCircle = L.circle([sLat, sLon], {
          radius: 250,
          color: "#f97316",
          fillColor: "#fb923c",
          fillOpacity: 0.25,
        }).addTo(map);

        vanMarker = L.circleMarker([vLat, vLon], {
          radius: 6,
          color: "#38bdf8",
          fillColor: "#0ea5e9",
          fillOpacity: 1.0,
        }).addTo(map);

        pathLatLngs = [[vLat, vLon]];
        vanPath = L.polyline(pathLatLngs, {
          weight: 3,
          opacity: 0.9,
        }).addTo(map);

        map.fitBounds([[sLat, sLon], [vLat, vLon]], { padding: [40, 40] });

        hideLoading();
      }

      if (msg.type === "van_update") {
        if (!map || !vanMarker) return;

        const lat = msg.lat;
        const lon = msg.lon;

        vanMarker.setLatLng([lat, lon]);
        pathLatLngs.push([lat, lon]);
        vanPath.setLatLngs(pathLatLngs);

        if (msg.distance_m !== undefined) {
          simDistEl.textContent = `${msg.distance_m.toFixed(1)} m`;
        }

        if (msg.status === "detected") {
          setStatus("detected");
        } else if (msg.status === "patrolling") {
          setStatus("patrolling");
        }
      }

      if (msg.type === "detection_result") {
        setStatus("idle");
        btnStart.disabled = false;
        btnReset.disabled = false;

        if (msg.monitoring) {
          monitoringLogEl.textContent = JSON.stringify(msg.monitoring, null, 2);

          if (msg.monitoring.model_version) {
            modelVersionEl.textContent = msg.monitoring.model_version;
          }
          if (msg.monitoring.latency_ms !== undefined) {
            latencyEl.textContent = msg.monitoring.latency_ms + " ms";
          }
          if (msg.monitoring.drift_score !== undefined) {
            driftScoreEl.textContent = msg.monitoring.drift_score;
          }
        }

        if (msg.registry && msg.registry.current_model_version) {
          modelVersionEl.textContent = msg.registry.current_model_version;
        }

        if (msg.forensic_bundle) {
          bundleNameEl.textContent = msg.forensic_bundle.bundle_name || "bundle";
          bundleLogEl.textContent = JSON.stringify(msg.forensic_bundle, null, 2);
        }
      }

      if (msg.type === "sim_end") {
        setStatus("idle");
        btnStart.disabled = false;
        btnReset.disabled = false;
      }

      if (msg.type === "error") {
        setStatus("idle");
        btnStart.disabled = false;
        btnReset.disabled = false;
        hideLoading();
        alert("Simulation error: " + (msg.message || "unknown error"));
      }
    };

    ws.onclose = () => {
      setStatus("idle");
      btnStart.disabled = false;
      btnReset.disabled = false;
    };
  }

  btnStart.addEventListener("click", async () => {
    showLoading();
    connectWebSocket();
    await fetch("/api/start_simulation", { method: "POST" });
  });

  btnReset.addEventListener("click", async () => {
    await fetch("/api/reset", { method: "POST" });
    setStatus("idle");
    btnStart.disabled = false;
    btnReset.disabled = true;
    resetGraphics();
    hideLoading();
  });

  ensureMap();
  connectWebSocket();
  hideLoading();
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

@app.post("/api/start_simulation")
async def start_simulation():
    if state.running:
        return JSONResponse({"status": "already_running"})
    # avvio loop in background
    asyncio.create_task(simulation_loop())
    return {"status": "started"}

@app.post("/api/reset")
def reset():
    state.running = False
    state.detected = False
    return {"status": "reset"}

@app.get("/api/status")
def api_status():
    monitoring = get_last_monitoring()
    registry = get_model_registry()
    bundle = get_last_forensic_bundle()
    return {
        "monitoring": monitoring,
        "registry": registry,
        "forensic_bundle": bundle,
        "running": state.running,
        "detected": state.detected,
        "current_simulation_id": state.current_sim_id,
    }

@app.post("/api/start_simulation_near")
async def start_simulation_near():
    if state.running:
        return JSONResponse({"status": "already_running"})

    asyncio.create_task(simulation_loop(force_near=True))
    return {"status": "started_debug"}


# ----------------------------------------------------
# WEBSOCKET
# ----------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_sockets.append(websocket)
    try:
        while True:
            # non ci aspettiamo messaggi dal client, ma dobbiamo await per tenere aperta la connessione
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_sockets:
            active_sockets.remove(websocket)
    except Exception:
        if websocket in active_sockets:
            active_sockets.remove(websocket)

# ----------------------------------------------------
# AVVIO LOCALE (opzionale)
# ----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ui_pention_m:app", host="0.0.0.0", port=8005, reload=True)
