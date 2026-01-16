# command: python validation/MLOps/mlops_pipeline_stress_test.py --n-runs 20
import os
import json
import time
import uuid
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import subprocess

DEFAULT_INGESTION_URL = "http://localhost:8011/ingest_data"
DEFAULT_METEO_URL = "http://localhost:8002/get_meteo"
LAT_MIN = 52.35
LAT_MAX = 52.39
LON_MIN = 4.88
LON_MAX = 4.92
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_CANDIDATES = [
    os.path.join(
        ROOT_DIR, "ClassificatoreNPS", "datasetNPS", "PENTION_EI_Complete.csv"
    ),
    os.path.join(ROOT_DIR, "test_datasetNPS", "PENTION_EI_Dataset.csv"),
]
DEFAULT_OUTPUT_CSV = os.path.join(
    ROOT_DIR, "validation", "MLOps", "mlops_stress_results.csv"
)
CONTAINERS_TO_MONITOR = [
    "mlops_ingestion",
    "mlops_monitoring",
    "gaussian_dispersion_model",
    "correction_dispersion_piml",
    "loc_emission_source_piml",
    "clas_nps",
]


def load_nps_dataset() -> Optional[pd.DataFrame]:
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
    if NPS_DF is None:
        return [0.0] * 600, "UNKNOWN"
    row = NPS_DF.sample(n=1).iloc[0]
    compound_name = row.get("Name", "UNKNOWN")
    try:
        s = row.iloc[1:601].values.astype(float).copy()
    except Exception:
        s = row.select_dtypes(include=[np.number]).values[:600].astype(float)
    shift = np.random.randint(-1, 2)
    s = np.roll(s, shift)
    drift = np.linspace(
        np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), len(s)
    )
    s += drift
    s = s * (1 + np.random.normal(0, 0.03, len(s)))
    dropout = np.random.rand(len(s)) < 0.02
    s[dropout] = 0
    s = np.clip(s, 0, None)
    s = s ** np.random.uniform(0.92, 1.05)
    s = np.clip(s, 0, 100)
    return s.tolist(), str(compound_name)


def random_latlon_in_bbox() -> Tuple[float, float]:
    lat = np.random.uniform(LAT_MIN, LAT_MAX)
    lon = np.random.uniform(LON_MIN, LON_MAX)
    return float(lat), float(lon)


def get_meteo(meteo_url: str) -> Dict[str, Any]:
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

    now_iso = datetime.utcnow().isoformat() + "Z"
    met = get_meteo(meteo_url)
    spectrum_noisy, true_compound = generate_noisy_spectrum()

    payload = {
        "simulation_id": sim_id,
        "timestamp": now_iso,
        "event_start_ts": now_iso,
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
            "noise_level": 0.08,
        },
        "SensorGPS": {"latitude": lat, "longitude": lon, "altitude_m": 2.0},
        "SourceGPS": {"latitude": source_lat, "longitude": source_lon},
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
    stats = {name: {"cpu": None, "mem": None} for name in containers}
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
        source_lat, source_lon = random_latlon_in_bbox()
        van_lat, van_lon = random_latlon_in_bbox()
        payload = build_simulation_payload(
            sim_id, van_lat, van_lon, source_lat, source_lon, meteo_url
        )
        if enable_docker_stats:
            docker_metrics_before = get_docker_stats(CONTAINERS_TO_MONITOR)
        else:
            docker_metrics_before = {
                name: {"cpu": None, "mem": None} for name in CONTAINERS_TO_MONITOR
            }
        t_start = time.time()
        resp = safe_post(ingestion_url, payload)
        t_end = time.time()
        e2e_latency_ms = round((t_end - t_start) * 1000.0, 2)
        code = resp.get("code", 0)
        body = resp.get("body", {})
        monitoring_block = None
        if isinstance(body, dict):
            monitoring_block = body.get("monitoring")

        latency_ms = None
        drift = None
        stability_index = None
        confidence = None
        model_version = None
        mse_free = None

        if isinstance(monitoring_block, dict):
            latency_ms = monitoring_block.get("latency_ms")
            drift = monitoring_block.get("drift_score")
            stability_index = monitoring_block.get("stability_index")
            confidence = monitoring_block.get("confidence")
            model_version = monitoring_block.get("model_version")
            mse_free = monitoring_block.get("mse_free")

        error_msg = None
        if code != 200:
            error_msg = json.dumps(body)[:500]

        row = {
            "run_id": i,
            "simulation_id": sim_id,
            "t_start_iso": datetime.utcfromtimestamp(t_start).isoformat() + "Z",
            "e2e_latency_ms": e2e_latency_ms,
            "latency_ms": latency_ms,
            "drift_score": drift,
            "stability_index": stability_index,
            "confidence": confidence,
            "model_version": model_version,
            "mse_free": mse_free,
            "http_status": code,
            "error_msg": error_msg,
            "van_lat": van_lat,
            "van_lon": van_lon,
            "source_lat": source_lat,
            "source_lon": source_lon,
        }

        for cname, m in docker_metrics_before.items():
            row[f"{cname}_cpu"] = m.get("cpu")
            row[f"{cname}_mem"] = m.get("mem")

        rows.append(row)
        print(
            f"[RUN {i}/{n_runs}] status={code}, latency={latency_ms}, drift={drift}, "
            f"conf={confidence}, version={model_version}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n[DONE] Stress test completed.\nSaved to:\n  {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stress test for the MLOps PENTION-M pipeline."
    )
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument(
        "--ingestion-url",
        type=str,
        default=os.environ.get("INGESTION_URL", DEFAULT_INGESTION_URL),
    )
    parser.add_argument(
        "--meteo-url", type=str, default=os.environ.get("METEO_URL", DEFAULT_METEO_URL)
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.environ.get("MLOPS_STRESS_CSV", DEFAULT_OUTPUT_CSV),
    )
    parser.add_argument("--no-docker-stats", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stress_test(
        n_runs=args.n_runs,
        ingestion_url=args.ingestion_url,
        meteo_url=args.meteo_url,
        output_csv=args.output_csv,
        enable_docker_stats=(
            not args.no - docker_stats
            if hasattr(args, "no-docker-stats")
            else not args.no_docker_stats
        ),
        seed=args.seed,
    )
