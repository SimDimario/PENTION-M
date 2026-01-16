import os
import json
import math
import statistics
from typing import Dict, Any
import pandas as pd

def safe_mean(values):
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else None

def safe_std(values):
    vals = [v for v in values if v is not None]
    if len(vals) <= 1:
        return None
    return float(statistics.pstdev(vals))

def load_piml_dispersion_kpi(piml_csv_path: str) -> Dict[str, Any]:
    """
    Read validation/PIML/validation_results_piml.csv and summarize:
    - rmse_free_mean / std
    - smoothness_mean / std
    - wind_alignment_mean / std
    - building_violation_ratio
    """
    if not os.path.exists(piml_csv_path):
        print(f"[WARN] PIML csv not found: {piml_csv_path}")
        return {}

    df = pd.read_csv(piml_csv_path)

    kpi = {
        "rmse_piml_mean": float(df["rmse_free"].mean()),
        "rmse_piml_std": float(df["rmse_free"].std()),
        "smoothness_mean": float(df["smoothness"].mean()),
        "smoothness_std": float(df["smoothness"].std()),
        "wind_alignment_mean": float(df["wind_alignment"].mean()),
        "wind_alignment_std": float(df["wind_alignment"].std()),
        "building_violation_ratio": float(df["building_violation"].mean()),
    }
    return kpi

def load_nps_kpi(nps_summary_path: str) -> Dict[str, Any]:
    """
    Read validation/NPS/results_nps/summary.json and summarize:
    - clf_brier_score
    - clf_conf_mean
    - clf_conf_std
    - clf_conf_range_min/max = mean ± std (clippato a [0,1])
    """
    if not os.path.exists(nps_summary_path):
        print(f"[WARN] NPS summary not found: {nps_summary_path}")
        return {}

    with open(nps_summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    brier = float(data.get("brier_score", 0.0))
    conf_mean = float(data.get("avg_confidence", 0.0))
    conf_std = float(data.get("std_confidence", 0.0))
    conf_min = max(0.0, conf_mean - conf_std)
    conf_max = min(1.0, conf_mean + conf_std)

    return {
        "clf_brier_score": brier,
        "clf_conf_mean": conf_mean,
        "clf_conf_std": conf_std,
        "clf_conf_range_min": conf_min,
        "clf_conf_range_max": conf_max,
    }

def load_localization_kpi(emission_metrics_path: str) -> Dict[str, Any]:
    """
    Read validation/Emission/emission_piml_metrics.json produced by the notebook
    emission_source_piml.ipynb:
    {
      "rmse_m": ...,
      "mae_m": ...
    }
    """
    if not os.path.exists(emission_metrics_path):
        print(f"[WARN] Emission metrics json not found: {emission_metrics_path}")
        return {}

    with open(emission_metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "localization_rmse_m": float(data.get("rmse_m", 0.0)),
        "localization_mae_m": float(data.get("mae_m", 0.0)),
    }

def load_mlops_kpi(monitoring_log_path: str) -> Dict[str, Any]:
    """
    Reads logs/monitoring_log.jsonl and calculates:
    - latency_ms_mean / median / max
    - drift_score_mean
    - drift_score_last
    - mse_free_mean
    """
    if not os.path.exists(monitoring_log_path):
        print(f"[WARN] Monitoring log not found: {monitoring_log_path}")
        return {}

    latencies = []
    drifts = []
    mses = []
    last_drift = None

    with open(monitoring_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            lat = obj.get("latency_ms")
            drift = obj.get("drift_score")
            mse = obj.get("mse_free")
            if isinstance(lat, (int, float)):
                latencies.append(float(lat))
            if isinstance(drift, (int, float)):
                drifts.append(float(drift))
                last_drift = float(drift)
            if isinstance(mse, (int, float)):
                mses.append(float(mse))

    if not latencies and not drifts and not mses:
        return {}
    kpi = {}
    if latencies:
        kpi["latency_ms_mean"] = float(sum(latencies) / len(latencies))
        kpi["latency_ms_median"] = float(statistics.median(latencies))
        kpi["latency_ms_max"] = float(max(latencies))
    if drifts:
        kpi["drift_score_mean"] = float(sum(drifts) / len(drifts))
        kpi["drift_score_last"] = last_drift
    if mses:
        kpi["mse_free_mean"] = float(sum(mses) / len(mses))
    return kpi

def load_forensic_kpi(forensic_csv_path: str) -> Dict[str, Any]:
    """
    Read validation/Forensic/forensic_validation_results.csv and calculate:
    - auditability_score = fraction of bundles with ALL True
    - share per column (hash_match, signature_ok, model_hash_match, map_hash_match)
    """
    if not os.path.exists(forensic_csv_path):
        print(f"[WARN] Forensic csv not found: {forensic_csv_path}")
        return {}

    df = pd.read_csv(forensic_csv_path)

    def col_ratio(col):
        vals = df[col].astype(str).str.lower()
        return float((vals == "true").mean())

    cols = ["hash_match", "signature_ok", "model_hash_match", "map_hash_match"]
    per_col = {f"forensic_{c}_ratio": col_ratio(c) for c in cols}
    all_ok = df[cols].astype(str).applymap(lambda x: x.lower() == "true")
    audit_score = float((all_ok.all(axis=1)).mean())

    return {
        "auditability_score": audit_score,
        **per_col,
        "forensic_num_bundles": int(len(df)),
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    piml_csv = os.path.join(script_dir, "PIML", "validation_results_piml.csv")
    nps_json = os.path.join(script_dir, "NPS", "results_nps", "summary.json")
    emission_json = os.path.join(script_dir, "Emission", "emission_piml_metrics.json")
    forensic_csv = os.path.join(script_dir, "Forensic", "forensic_validation_results.csv")
    monitoring_log = os.path.join(root_dir, "logs", "monitoring_log.jsonl")
    kpi = {}
    kpi.update(load_piml_dispersion_kpi(piml_csv))
    kpi.update(load_localization_kpi(emission_json))
    kpi.update(load_nps_kpi(nps_json))
    kpi.update(load_mlops_kpi(monitoring_log))
    kpi.update(load_forensic_kpi(forensic_csv))
    kpi["system_version"] = "PENTION-M_PIML_v1"
    df = pd.DataFrame([kpi])
    out_csv = os.path.join(script_dir, "final_kpi_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved final KPI table to: {out_csv}\n")
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df)

if __name__ == "__main__":
    main()