import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, random_split

# === Import locali PIML ===
from MCxM_PIML import MCxM_PIML
from loss_function_piml import physics_masked_loss_piml
from CNNDataset import CNNDataset2

# ============================================================
# CONFIGURAZIONE BASE
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "mcxm_piml_model_best.pth")
MODEL_PATH = "MODEL_NOT_SAVED_IN_FAST_RETRAIN.pth"
LOGS_DIR = "/logs"
REGISTRY_PATH = os.path.join(LOGS_DIR, "model_registry.json")

# Dataset
BINARY_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data", "amsterdam_netherlands_bbox.npy")
REAL_CONC_PATH = os.path.join(SCRIPT_DIR, "dataset", "real_dispersion")
CSV_PATH = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-11-24_PIML_processed.csv")

# ============================================================
# FUNZIONE DI RETRAINING REALE
# ============================================================

def retrain_model():
    """
    Esegue un retraining PIML rapido (10 epoche, tutte le mappe).
    Ritorna (new_version, metrics) per il registry MLOps.
    NON scrive direttamente nel registry: ci pensa mock_retrain.
    """
    print("[RetrainService] 🔁 Avvio retraining PIML reale...")

    # --- Caricamento risorse base ---
    if not os.path.exists(BINARY_MAP_PATH):
        raise FileNotFoundError(f"Binary map non trovata: {BINARY_MAP_PATH}")
    if not os.path.exists(REAL_CONC_PATH):
        raise FileNotFoundError(f"Cartella real_dispersion non trovata: {REAL_CONC_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV dataset non trovato: {CSV_PATH}")

    binary_map = np.load(BINARY_MAP_PATH)
    m = binary_map.shape[0]

    csv_df = pd.read_csv(CSV_PATH)
    if "simulation_id" not in csv_df.columns:
        raise ValueError("CSV PIML: manca la colonna 'simulation_id'")

    csv_df_reduced = csv_df.groupby("simulation_id").first().reset_index()
    needed_cols = ["wind_dir_cos", "wind_dir_sin", "wind_speed", "gps_x", "gps_y"]
    for c in needed_cols:
        if c not in csv_df_reduced.columns:
            raise ValueError(f"CSV PIML: manca la colonna '{c}'")

    csv_df_reduced = csv_df_reduced[needed_cols]

    # --- Caricamento mappe reali ---
    concentration_maps, wind_dirs, wind_speeds = [], [], []

    files = sorted(f for f in os.listdir(REAL_CONC_PATH) if f.endswith(".npy"))
    # Usa solo le prime 30 mappe per retrain veloce (prima erano ~300)
    files = files[:30]
    if not files:
        raise RuntimeError(f"Nessuna mappa .npy trovata in {REAL_CONC_PATH}")

    for file in files:
        fpath = os.path.join(REAL_CONC_PATH, file)
        conc_map = np.load(fpath)

        # Supporto mappe 2D (nuovo modello PIML)
        if conc_map.ndim == 2:
            conc_map_mean = conc_map.astype(np.float32)

        # Supporto retrocompatibile mappe 3D (snapshot finale)
        elif conc_map.ndim == 3:
            conc_map_mean = conc_map[:, :, -1].astype(np.float32)

        else:
            raise ValueError(f"conc_map {file} shape inattesa: {conc_map.shape}")

        try:
            i = int(file.split("_")[1])
        except Exception:
            print(f"[RetrainService] [WARN] Impossibile ricavare indice da filename: {file}")
            continue

        if i >= len(csv_df_reduced):
            print(f"[RetrainService] [WARN] Indice {i} fuori range CSV (len={len(csv_df_reduced)}), skip {file}")
            continue

        wind_dir_cos, wind_dir_sin, wind_speed, gps_x, gps_y = csv_df_reduced.iloc[i]
        rad_angle = np.arctan2(wind_dir_sin, wind_dir_cos)
        degree_angle = np.degrees(rad_angle)

        concentration_maps.append(conc_map_mean)
        wind_dirs.append(degree_angle)
        wind_speeds.append(wind_speed)

    n_maps = len(concentration_maps)
    print(f"[RetrainService] [DEBUG] Dataset costruito con {n_maps} mappe valide.")

    if n_maps < 10:
        print(f"[RetrainService] [WARN] Solo {n_maps} mappe valide. Il retrain potrebbe essere poco stabile.")

    if n_maps == 0:
        raise RuntimeError("Nessuna mappa valida per il retrain PIML.")

    # --- Normalizzazione e costruzione dataset ---
    dataset = CNNDataset2(
        concentration_maps,
        wind_dirs,
        wind_speeds,
        global_features=None,
        m=m
    )

    n_total = len(dataset)
    n_train = max(1, int(0.8 * n_total))
    n_val = max(1, n_total - n_train)
    if n_train + n_val > n_total:
        n_val = n_total - n_train

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RetrainService] [INFO] Device: {device}")

    # --- Modello ---
    model = MCxM_PIML(binary_map, m=m, n_channel=1, wind_dim=2, n_global_features=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float("inf")

    from time import time
    t_start = time()

    epochs = 2
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for conc_map, wind_dir, wind_speed, _ in train_loader:
            conc_map = conc_map.to(device)      # [B,1,m,m]
            wind_dir = wind_dir.to(device)      # [B]
            wind_speed = wind_speed.to(device)  # [B]

            optimizer.zero_grad()
            output = model(
                conc_map,
                torch.stack([wind_speed, wind_dir], dim=1)
            )

            wind_vec = (
                torch.cos(torch.deg2rad(wind_dir)).mean().item(),
                torch.sin(torch.deg2rad(wind_dir)).mean().item(),
            )

            target = conc_map.squeeze(1)  # [B,m,m]
            loss, comps = physics_masked_loss_piml(
                output,
                target,
                binary_map,
                wind_vector=wind_vec
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for conc_map, wind_dir, wind_speed, _ in val_loader:
                conc_map = conc_map.to(device)
                wind_dir = wind_dir.to(device)
                wind_speed = wind_speed.to(device)

                output = model(
                    conc_map,
                    torch.stack([wind_speed, wind_dir], dim=1),
                )

                wind_vec = (
                    torch.cos(torch.deg2rad(wind_dir)).mean().item(),
                    torch.sin(torch.deg2rad(wind_dir)).mean().item(),
                )

                target = conc_map.squeeze(1)  # [B,m,m]
                v_loss, comps = physics_masked_loss_piml(
                    output,
                    target,
                    binary_map,
                    wind_vector=wind_vec
                )
                val_loss += v_loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(
            f"[RetrainService] Epoch {epoch+1}/{epochs} "
            f"- Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}"
        )

        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), MODEL_PATH)
        #     print(f"[RetrainService] Nuovo best model salvato in {MODEL_PATH} (val_loss={best_val_loss:.6f})")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"[RetrainService] Best model *virtuale* aggiornato (val_loss={best_val_loss:.6f})")

    duration_min = round((time() - t_start) / 60, 2)
    print(f"[RetrainService] Durata totale retraining: {duration_min} minuti")

    # --- Versioning numerico incrementale ---
    os.makedirs(LOGS_DIR, exist_ok=True)
    prev_version = None
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                prev_version = json.load(f).get("current_model_version")
        except Exception:
            prev_version = None

    if prev_version and isinstance(prev_version, str) and prev_version.startswith("PIML_v"):
        try:
            i = int(prev_version.replace("PIML_v", ""))
            new_version = f"PIML_v{i+1}"
        except Exception:
            new_version = "PIML_v1"
    else:
        new_version = "PIML_v1"

    metrics = {
        "final_val_loss": float(round(best_val_loss, 6)),
        "samples": int(len(dataset)),
        "epochs": int(epochs),
        "duration_min": float(duration_min),
        # "model_path": MODEL_PATH,
        "model_path": "NOT_SAVED_FAST_RETRAIN",

    }

    print(f"[RetrainService] Retraining completato. new_version={new_version}, metrics={metrics}")

    return new_version, metrics

# ============================================================
# TEST MANUALE (locale nel container)
# ============================================================

if __name__ == "__main__":
    retrain_model()
