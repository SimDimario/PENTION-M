# comando: python validation/MLOps/mlops_pipeline_stress_test.py --n-runs 20

import os
import json
import time
import uuid
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import subprocess

# ---------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------

# URL di default dell'ingestion service (host locale)
DEFAULT_INGESTION_URL = "http://localhost:8011/ingest_data"

# URL opzionale per ottenere meteo da gaussian_dispersion_model
DEFAULT_METEO_URL = "http://localhost:8002/get_meteo"

# Bounding box usata in api_ingestion.py per Amsterdam (lat/lon)
LAT_MIN, LAT_MAX = 52.35, 52.39
LON_MIN, LON_MAX = 4.88, 4.92

# Percorso dataset NPS (spettro EI)
# Si tenta prima quello usato da UI, poi una fallback in test_datasetNPS
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_CANDIDATES = [
    os.path.join(ROOT_DIR, "ClassificatoreNPS", "datasetNPS", "PENTION_EI_Complete.csv"),
    os.path.join(ROOT_DIR, "test_datasetNPS", "PENTION_EI_Dataset.csv"),
]

# Output CSV di default
DEFAULT_OUTPUT_CSV = os.path.join(
    ROOT_DIR, "validation", "MLOps", "mlops_stress_results.csv"
)

# Container da monitorare (se docker stats è disponibile)
CONTAINERS_TO_MONITOR = [
    "mlops_ingestion",
    "mlops_monitoring",
    "gaussian_dispersion_model",
    "correction_dispersion_piml",
    "loc_emission_source_piml",
    "clas_nps",
]


# ---------------------------------------------------
# UTILITIES
# ---------------------------------------------------

def load_nps_dataset() -> Optional[pd.DataFrame]:
    """Carica un dataset NPS compatibile con lo spettro EI 1–600."""
    for path in DATASET_CANDIDATES:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"[INFO] NPS dataset loaded from: {path}")
                return df
            except Exception as e:
                print(f"[WARN] Error loading NPS dataset from {path}: {e}")
    print("[WARN] No NPS dataset found; spectra will be zeros.")
    return None


NPS_DF = load_nps_dataset()


def generate_noisy_spectrum(noise_level: float = 0.08) -> Tuple[List[float], str]:

    """
    Replica la logica di UI_Pention_M:
    - sceglie una molecola a caso
    - applica jitter, rumore, drift di baseline, dropout, scaling non lineare
    Ritorna (spettro_600, compound_name).
    """
    if NPS_DF is None:
        return [0.0] * 600, "UNKNOWN"

    row = NPS_DF.sample(n=1).iloc[0]

    # Si assume che le colonne 1..600 siano lo spettro EI (come in UI)
    compound_name = row.get("Name", "UNKNOWN")
    try:
        s = row.iloc[1:601].values.astype(float).copy()
    except Exception:
        # Fallback: prima riga di soli numeri
        s = row.select_dtypes(include=[np.number]).values[:600].astype(float)

    # (1) jitter ±1 m/z
    shift = np.random.randint(-1, 2)
    if shift != 0:
        s = np.roll(s, shift)

    # (2) baseline drift
    drift = np.linspace(
        np.random.uniform(-0.4, 0.4),
        np.random.uniform(-0.4, 0.4),
        len(s)
    )
    s = s + drift

    # (3) multiplicative noise
    s = s * (1 + np.random.normal(0, 0.03, len(s)))

    # (4) peak dropout
    dropout = np.random.rand(len(s)) < 0.02
    s[dropout] = 0

    # (4.5) nothing negative before power
    s = np.clip(s, 0, None)

    # (5) non-linear scaling
    s = s ** np.random.uniform(0.92, 1.05)

    # (6) clipping stile EI
    s = np.clip(s, 0, 100)

    return s.tolist(), str(compound_name)


def random_latlon_in_bbox() -> Tuple[float, float]:
    lat = np.random.uniform(LAT_MIN, LAT_MAX)
    lon = np.random.uniform(LON_MIN, LON_MAX)
    return float(lat), float(lon)


def get_meteo(meteo_url: str) -> Dict[str, Any]:
    """
    Prova a interrogare gaussian_dispersion_model per ottenere meteo.
    In caso di errore, usa valori di fallback fisicamente plausibili.
    """
    try:
        r = requests.get(meteo_url, timeout=5)
        r.raise_for_status()
        data = r.json()
        return {
            "temperature_C": float(data.get("temperature", 20.0)),
            "humidity_%": float(data.get("humidity", 0.5)),
            "wind_speed_mps": float(data.get("wind_speed", 4.0)),
            "wind_dir_deg": int(data.get("wind_dir_deg", 180)),
            "stability_class": str(data.get("stability_class", "C")),
        }
    except Exception as e:
        print(f"[WARN] Meteo service not reachable ({e}), using fallback values.")
        return {
            "temperature_C": 20.0,
            "humidity_%": 0.5,
            "wind_speed_mps": 4.0,
            "wind_dir_deg": 180,
            "stability_class": "C",
        }


def build_simulation_payload(
    sim_id: str,
    lat: float,
    lon: float,
    source_lat: float,
    source_lon: float,
    meteo_url: str,
) -> Dict[str, Any]:
    """
    Costruisce un payload identico alla UI (ui_pention_m.build_simulation_payload),
    usando meteo fisico (se possibile) e spettro EI simulato.
    """
    now_iso = datetime.utcnow().isoformat() + "Z"

    # Meteo fisico o fallback
    met = get_meteo(meteo_url)

    # Spettro EI simulato
    noise_level = 0.08
    spectrum_noisy, true_compound = generate_noisy_spectrum(noise_level=noise_level)

    payload = {
        "simulation_id": sim_id,
        "timestamp": now_iso,
        "SensorAir": {
            "temperature_C": met["temperature_C"],
            "humidity_%": met["humidity_%"],
            "wind_speed_mps": met["wind_speed_mps"],
            "wind_dir_deg": met["wind_dir_deg"],
            "stability_class": met["stability_class"],
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
        "SourceGPS": {
            "latitude": source_lat,
            "longitude": source_lon,
        },
    }

    return payload


def safe_post(url: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        code = r.status_code
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text}
        return {"code": code, "body": body}
    except Exception as e:
        return {"code": 500, "body": {"error": str(e)}}


def get_docker_stats(containers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Prova a usare `docker stats --no-stream` per recuperare CPU% e MEM%.
    Se docker non è disponibile o il comando fallisce, ritorna valori None.
    """
    stats: Dict[str, Dict[str, Optional[float]]] = {
        name: {"cpu": None, "mem": None} for name in containers
    }

    try:
        cmd = [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.Name}},{{.CPUPerc}},{{.MemPerc}}",
        ] + containers
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        print(f"[WARN] docker stats not available ({e}), skipping resource metrics.")
        return stats

    for line in output.strip().splitlines():
        try:
            name, cpu_s, mem_s = line.split(",")
            cpu = float(cpu_s.replace("%", "").strip())
            mem = float(mem_s.replace("%", "").strip())
            if name in stats:
                stats[name]["cpu"] = cpu
                stats[name]["mem"] = mem
        except Exception:
            continue

    return stats


# ---------------------------------------------------
# MAIN STRESS TEST
# ---------------------------------------------------

def run_stress_test(
    n_runs: int,
    ingestion_url: str,
    meteo_url: str,
    output_csv: str,
    enable_docker_stats: bool = True,
    seed: int = 42,
):
    np.random.seed(seed)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rows = []

    for i in range(1, n_runs + 1):
        sim_id = f"STRESS_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{i:03d}"

        # Sorgente e posizione del van scelti casualmente nella bounding box
        source_lat, source_lon = random_latlon_in_bbox()
        van_lat, van_lon = random_latlon_in_bbox()

        payload = build_simulation_payload(
            sim_id=sim_id,
            lat=van_lat,
            lon=van_lon,
            source_lat=source_lat,
            source_lon=source_lon,
            meteo_url=meteo_url,
        )

        # Metriche risorse (opzionali)
        if enable_docker_stats:
            docker_metrics_before = get_docker_stats(CONTAINERS_TO_MONITOR)
        else:
            docker_metrics_before = {name: {"cpu": None, "mem": None} for name in CONTAINERS_TO_MONITOR}

        t_start = time.time()
        resp = safe_post(ingestion_url, payload)
        t_end = time.time()

        e2e_latency_ms = round((t_end - t_start) * 1000.0, 2)

        code = resp.get("code", 0)
        body = resp.get("body", {})

        # Estraggo blocco monitoring dalla risposta api_ingestion
        monitoring_block = None
        if isinstance(body, dict):
            monitoring_block = body.get("monitoring")

        drift = None
        svc_latency_ms = None
        stability_index = None
        confidence = None
        model_version = None

        if isinstance(monitoring_block, dict):
            drift = monitoring_block.get("drift_score")
            svc_latency_ms = monitoring_block.get("latency_ms")
            stability_index = monitoring_block.get("stability_index")
            confidence = monitoring_block.get("confidence")
            model_version = monitoring_block.get("model_version")

        # mse_free viene inviato solo nel forensic/monitoring payload,
        # ma dal servizio monitoring risulta nel log; qui usiamo None e lo analizziamo dal log in notebook.
        mse_free = None

        error_msg = None
        if code != 200:
            error_msg = json.dumps(body)[:500]

        # Metriche risorse dopo la chiamata (semplice: usiamo solo "before")
        row = {
            "run_id": i,
            "simulation_id": sim_id,
            "t_start_iso": datetime.utcfromtimestamp(t_start).isoformat() + "Z",
            "e2e_latency_ms": e2e_latency_ms,
            "svc_latency_ms": svc_latency_ms,
            "drift_score": drift,
            "mse_free": mse_free,
            "stability_index": stability_index,
            "confidence": confidence,
            "model_version": model_version,
            "http_status": code,
            "error_msg": error_msg,
            "van_lat": van_lat,
            "van_lon": van_lon,
            "source_lat": source_lat,
            "source_lon": source_lon,
        }

        # Aggiungiamo CPU/MEM per alcuni container (se presenti)
        for cname, m in docker_metrics_before.items():
            row[f"{cname}_cpu"] = m.get("cpu")
            row[f"{cname}_mem"] = m.get("mem")

        rows.append(row)

        print(
            f"[RUN {i}/{n_runs}] status={code}, e2e_ms={e2e_latency_ms}, "
            f"svc_ms={svc_latency_ms}, drift={drift}, model={model_version}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n[DONE] Stress test completed. Results saved to:\n  {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stress test per la pipeline MLOps PENTION-M (ingestion + monitoring + PIML)."
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=20,
        help="Numero di simulazioni consecutive da eseguire (default: 20).",
    )
    parser.add_argument(
        "--ingestion-url",
        type=str,
        default=os.environ.get("INGESTION_URL", DEFAULT_INGESTION_URL),
        help=f"URL del servizio ingestion (default: {DEFAULT_INGESTION_URL}).",
    )
    parser.add_argument(
        "--meteo-url",
        type=str,
        default=os.environ.get("METEO_URL", DEFAULT_METEO_URL),
        help=f"URL del servizio meteo gaussian_dispersion_model (default: {DEFAULT_METEO_URL}).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.environ.get("MLOPS_STRESS_CSV", DEFAULT_OUTPUT_CSV),
        help=f"Percorso file CSV di output (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--no-docker-stats",
        action="store_true",
        help="Disabilita la raccolta di metriche da `docker stats`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per la riproducibilità (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stress_test(
        n_runs=args.n_runs,
        ingestion_url=args.ingestion_url,
        meteo_url=args.meteo_url,
        output_csv=args.output_csv,
        enable_docker_stats=not args.no_docker_stats,
        seed=args.seed,
    )
