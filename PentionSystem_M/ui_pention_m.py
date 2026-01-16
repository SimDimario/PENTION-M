import asyncio
import json
import os
import random
from datetime import datetime
from glob import glob
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse, JSONResponse
import networkx as nx
import osmnx as ox
import sys

sys.path.append("/shared_config")
from config_geo import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

AMSTERDAM_PLACE = "Amsterdam, Netherlands"
CACHE_DIR = "/app/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
GRAPH_PATH = os.path.join(CACHE_DIR, "amsterdam_drive.graphml")
DETECTION_RADIUS_M = 300.0
STEP_DELAY_SEC = 0.18
LOG_DIR = "/logs"
INGESTION_URL = "http://mlops_ingestion:8011/ingest_data"
app = FastAPI(title="PENTION-M UI (Van Simulation)")
app.mount("/static", StaticFiles(directory="static"), name="static")
DATASET_PATH = "/app/datasetNPS/PENTION_EI_Complete.csv"

try:
    NPS_DF = pd.read_csv(DATASET_PATH)
    print("[UI] NPS dataset loaded for spectrum generation.", flush=True)
except Exception as e:
    print(f"[UI] ERROR loading NPS dataset: {e}", flush=True)
    NPS_DF = None


class SimulationState:
    def __init__(self):
        self.G: nx.MultiDiGraph | None = None
        self.source_node = None
        self.van_node = None
        self.running = False
        self.detected = False
        self.path = []
        self.current_sim_id = None


state = SimulationState()
active_sockets: list[WebSocket] = []


def load_graph():
    """Loads the Amsterdam road graph or, as a fallback, a synthetic grid graph."""
    if state.G is not None:
        return state.G

    G = None

    if os.path.exists(GRAPH_PATH):
        try:
            print("[UI] Loading graph from cache: ", GRAPH_PATH, flush=True)
            G = ox.load_graphml(GRAPH_PATH)
            G = nx.Graph(G)

            lats = [v["y"] for k, v in G.nodes(data=True)]
            lons = [v["x"] for k, v in G.nodes(data=True)]
            print(
                "[UI] BOUNDING BOX REAL:",
                "lat:",
                min(lats),
                max(lats),
                "lon:",
                min(lons),
                max(lons),
                flush=True,
            )

        except Exception as e:
            print(f"[UI] Error loading cached graph: {e}", flush=True)
            G = None

    if G is None:
        try:
            print("[UI] Downloading graph from OpenStreetMap...", flush=True)
            G = ox.graph_from_place(AMSTERDAM_PLACE, network_type="drive")
            ox.save_graphml(G, GRAPH_PATH)
            G = nx.Graph(G)
            print("[UI] Graph downloaded and cached.", flush=True)
        except Exception as e:
            print(
                f"[UI] ERROR downloading graph, falling back to synthetic grid: {e}",
                flush=True,
            )
            G = None

    if G is None:
        print("[UI] Building synthetic grid graph for Amsterdam area.", flush=True)
        G = nx.grid_2d_graph(20, 20)

        for i, j in G.nodes:
            fi = i / 19.0
            fj = j / 19.0
            lat = LAT_MIN + (LAT_MAX - LAT_MIN) * fi
            lon = LON_MIN + (LON_MAX - LON_MIN) * fj
            G.nodes[(i, j)]["y"] = float(lat)
            G.nodes[(i, j)]["x"] = float(lon)

    state.G = G
    return G


def node_latlon(G, node):
    data = G.nodes[node]
    return float(data["y"]), float(data["x"])


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


async def broadcast(message: dict):
    """Sends a JSON message to all connected WebSocket clients."""
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
    """Reads the latest entry from monitoring_log.jsonl, if it exists."""
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
    """Gets the latest JSON forensic bundle in /logs/forensic."""
    dir_forensic = os.path.join(LOG_DIR, "forensic")
    if not os.path.isdir(dir_forensic):
        return None
    files = sorted(
        glob(os.path.join(dir_forensic, "bundle_*.json")), key=os.path.getmtime
    )
    if not files:
        return None
    try:
        with open(files[-1], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def generate_noisy_spectrum(noise_level: float):
    if NPS_DF is None:
        return [0.0] * 600, "UNKNOWN"

    row = NPS_DF.sample(n=1).iloc[0]
    compound_name = row["Name"]

    s = row.iloc[1:601].values.astype(float).copy()

    shift = np.random.randint(-1, 2)
    if shift != 0:
        s = np.roll(s, shift)

    drift = np.linspace(
        np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), len(s)
    )
    s = s + drift
    s = s * (1 + np.random.normal(0, 0.03, len(s)))
    dropout = np.random.rand(len(s)) < 0.02
    s[dropout] = 0
    s = np.clip(s, 0, None)
    s = s ** np.random.uniform(0.92, 1.05)
    s = np.clip(s, 0, 100)

    return s.tolist(), compound_name


def build_simulation_payload(
    sim_id: str, lat: float, lon: float, source_lat: float, source_lon: float
):
    """
    NEW VERSION — The UI only sends raw data.
    Physical weather from GaussianPuff /get_meteo.
    """
    now_iso = datetime.utcnow().isoformat() + "Z"

    try:
        resp_met = requests.get(
            "http://gaussian_dispersion_model:8002/get_meteo", timeout=10
        ).json()
        temperature = resp_met.get("temperature", 20.0)
        humidity = resp_met.get("humidity", 0.5)
        wind_speed = resp_met.get("wind_speed", 4.0)
        wind_dir_deg = resp_met.get("wind_dir_deg", 180)
        stability_class = resp_met.get("stability_class", "C")
    except Exception:
        temperature = 20.0
        humidity = 0.5
        wind_speed = 4.0
        wind_dir_deg = 180
        stability_class = "C"

    noise_level = 0.08
    spectrum_noisy, true_compound = generate_noisy_spectrum(noise_level)

    payload = {
        "simulation_id": sim_id,
        "timestamp": now_iso,
        "SensorAir": {
            "temperature_C": temperature,
            "humidity_%": humidity,
            "wind_speed_mps": wind_speed,
            "wind_dir_deg": wind_dir_deg,
            "stability_class": stability_class,
        },
        "SensorSubstance": {
            "compound_name": true_compound,
            "concentration_series_mg_m3": spectrum_noisy,
            "unit": "intensity",
            "noise_level": noise_level,
        },
        "SensorGPS": {
            "latitude": lat,
            "longitude": lon,
            "altitude_m": 2.0,
        },
        "SourceGPS": {"latitude": source_lat, "longitude": source_lon},
    }

    return payload


def call_ingestion_pipeline(
    sim_id: str, lat: float, lon: float, source_lat: float, source_lon: float
):
    payload = build_simulation_payload(sim_id, lat, lon, source_lat, source_lon)

    payload["event_start_ts"] = datetime.utcnow().isoformat() + "Z"

    try:
        resp = requests.post(INGESTION_URL, json=payload, timeout=180)
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return {"code": resp.status_code, "body": body}
    except Exception as e:
        return {"code": 500, "body": {"error": str(e)}}


async def simulation_loop(force_near=False):
    try:
        G = load_graph()

        state.running = True
        state.detected = False
        state.path = []

        nodes = list(G.nodes)
        if not nodes:
            raise RuntimeError("Graph has no nodes – cannot start simulation.")

        state.source_node = random.choice(nodes)
        source_lat, source_lon = node_latlon(G, state.source_node)

        if force_near:
            candidates = []
            for n in nodes:
                lat, lon = node_latlon(G, n)
                dist = haversine_m(lat, lon, source_lat, source_lon)
                if 1800 < dist < 2000:
                    candidates.append(n)

            if candidates:
                state.van_node = random.choice(candidates)
            else:
                neighbors = list(G.neighbors(state.source_node))
                if neighbors:
                    state.van_node = random.choice(neighbors)
                else:
                    state.van_node = random.choice(nodes)
        else:
            while True:
                candidate = random.choice(nodes)
                if candidate != state.source_node:
                    state.van_node = candidate
                    break

        van_lat, van_lon = node_latlon(G, state.van_node)
        state.path.append((van_lat, van_lon))
        sim_id = f"SIM_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        state.current_sim_id = sim_id
        await broadcast(
            {
                "type": "init",
                "simulation_id": sim_id,
                "source": {"lat": source_lat, "lon": source_lon},
                "van": {"lat": van_lat, "lon": van_lon},
                "status": "patrolling",
            }
        )
        if force_near:
            try:
                path_nodes = nx.shortest_path(
                    G, source=state.van_node, target=state.source_node, weight="length"
                )
            except Exception:
                path_nodes = [state.van_node, state.source_node]

            for pn in path_nodes[1:]:
                if not state.running:
                    break

                state.van_node = pn
                lat, lon = node_latlon(G, pn)
                state.path.append((lat, lon))

                dist = haversine_m(lat, lon, source_lat, source_lon)

                await broadcast(
                    {
                        "type": "van_update",
                        "lat": lat,
                        "lon": lon,
                        "status": "patrolling",
                        "distance_m": dist,
                    }
                )

                INNER_RADIUS = DETECTION_RADIUS_M - 50
                if dist <= INNER_RADIUS:
                    state.detected = True
                    state.running = False

                    await broadcast(
                        {
                            "type": "van_update",
                            "lat": lat,
                            "lon": lon,
                            "status": "detected",
                            "distance_m": dist,
                        }
                    )
                    await asyncio.sleep(0.1)

                    break

                await asyncio.sleep(STEP_DELAY_SEC)

            if state.detected:
                result = call_ingestion_pipeline(
                    sim_id, lat, lon, source_lat, source_lon
                )
                monitoring = None
                if isinstance(result.get("body"), dict):
                    monitoring = result["body"].get("monitoring")
                if monitoring is None:
                    monitoring = get_last_monitoring()
                registry = get_model_registry()
                bundle = get_last_forensic_bundle()
                await broadcast(
                    {
                        "type": "detection_result",
                        "simulation_id": sim_id,
                        "ingestion_response": result,
                        "monitoring": monitoring,
                        "registry": registry,
                        "forensic_bundle": bundle,
                    }
                )

            return

        while state.running and not state.detected:
            neighbors = list(G.neighbors(state.van_node))
            if not neighbors:
                break

            state.van_node = random.choice(neighbors)
            van_lat, van_lon = node_latlon(G, state.van_node)
            state.path.append((van_lat, van_lon))
            dist = haversine_m(van_lat, van_lon, source_lat, source_lon)
            INNER_RADIUS = DETECTION_RADIUS_M - 50

            if dist <= INNER_RADIUS:
                state.detected = True
                state.running = False

                await broadcast(
                    {
                        "type": "van_update",
                        "lat": van_lat,
                        "lon": van_lon,
                        "status": "detected",
                        "distance_m": dist,
                    }
                )

                await asyncio.sleep(0.15)

                result = call_ingestion_pipeline(
                    sim_id, van_lat, van_lon, source_lat, source_lon
                )
                monitoring = (
                    result.get("body", {}).get("monitoring") or get_last_monitoring()
                )
                registry = get_model_registry()
                bundle = get_last_forensic_bundle()

                await broadcast(
                    {
                        "type": "detection_result",
                        "simulation_id": sim_id,
                        "ingestion_response": result,
                        "monitoring": monitoring,
                        "registry": registry,
                        "forensic_bundle": bundle,
                    }
                )

                break

            await broadcast(
                {
                    "type": "van_update",
                    "lat": van_lat,
                    "lon": van_lon,
                    "status": "patrolling",
                    "distance_m": dist,
                }
            )

            await asyncio.sleep(STEP_DELAY_SEC)

        if not state.detected:
            await broadcast(
                {
                    "type": "sim_end",
                    "simulation_id": sim_id,
                    "reason": "stopped_or_completed",
                }
            )

    except Exception as e:
        state.running = False
        state.detected = False
        print(f"[UI] simulation_loop error: {e}", flush=True)
        await broadcast(
            {
                "type": "error",
                "message": str(e),
            }
        )


@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse("static/index.html")


@app.post("/api/start_simulation")
async def start_simulation():
    if state.running:
        return JSONResponse({"status": "already_running"})
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_sockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_sockets:
            active_sockets.remove(websocket)
    except Exception:
        if websocket in active_sockets:
            active_sockets.remove(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ui_pention_m:app", host="0.0.0.0", port=8005, reload=True)
