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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
DETECTION_RADIUS_M = 300.0  # raggio operativo per "detection"
STEP_DELAY_SEC = 0.5        # tempo tra un passo e l'altro
LOG_DIR = "/logs"
INGESTION_URL = "http://mlops_ingestion:8011/ingest_data"

app = FastAPI(title="PENTION-M UI (Van Simulation)")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
                if 1000 < dist < 1400:
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

                    await broadcast({
                        "type": "van_update",
                        "lat": lat,
                        "lon": lon,
                        "status": "detected",
                        "distance_m": dist
                    })
                    await asyncio.sleep(0.1)

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

@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse("static/index.html")

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
