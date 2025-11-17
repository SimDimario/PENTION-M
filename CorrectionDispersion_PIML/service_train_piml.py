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
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "mcxm_piml_model_best.pth")
LOGS_DIR = "/logs"
REGISTRY_PATH = os.path.join(LOGS_DIR, "model_registry.json")

# Dataset
BINARY_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data/amsterdam_netherlands_bbox.npy")
REAL_CONC_PATH = os.path.join(SCRIPT_DIR, "dataset", "real_dispersion")
CSV_PATH = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-11-10_PIML_processed.csv")

# ============================================================
# FUNZIONE DI RETRAINING REALE
# ============================================================

def retrain_model():
    """
    Esegue un retraining PIML rapido (10 epoche, tutte le mappe).
    Ritorna (new_version, metrics) per il registry MLOps.
    """
    print("[RetrainService] 🔁 Avvio retraining PIML reale...")

    # --- Caricamento risorse base ---
    binary_map = np.load(BINARY_MAP_PATH)
    m = binary_map.shape[0]

    csv_df = pd.read_csv(CSV_PATH)
    csv_df_reduced = csv_df.groupby("simulation_id").first().reset_index()
    csv_df_reduced = csv_df_reduced[["wind_dir_cos", "wind_dir_sin", "wind_speed", "gps_x", "gps_y"]]

    # --- Caricamento mappe reali ---
    concentration_maps, wind_dirs, wind_speeds = [], [], []
    for file in sorted(os.listdir(REAL_CONC_PATH)):
        if not file.endswith(".npy"):
            continue
        conc_map = np.load(os.path.join(REAL_CONC_PATH, file))
        conc_map_mean = np.mean(conc_map, axis=2)
        i = int(file.split("_")[1])
        if i >= len(csv_df_reduced):
            continue

        wind_dir_cos, wind_dir_sin, wind_speed, gps_x, gps_y = csv_df_reduced.iloc[i]
        rad_angle = np.arctan2(wind_dir_sin, wind_dir_cos)
        degree_angle = np.degrees(rad_angle)

        concentration_maps.append(conc_map_mean)
        wind_dirs.append(degree_angle)
        wind_speeds.append(wind_speed)

    gps_features = list(zip(csv_df_reduced["gps_x"], csv_df_reduced["gps_y"]))

    print(f"[DEBUG] Dataset costruito con {len(concentration_maps)} mappe valide.")

    # --- Normalizzazione e costruzione dataset ---
    dataset = CNNDataset2(concentration_maps, wind_dirs, wind_speeds, global_features=gps_features, m=m)
    train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RetrainService] [INFO] Device: {device}")

    # --- Modello ---
    model = MCxM_PIML(binary_map, m=m, n_channel=1, wind_dim=2, n_global_features=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float("inf")

    # === ⏱️ MISURAZIONE TEMPO DI TRAINING ===
    from time import time
    t_start = time()

    # --- Training completo ---
    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for conc_map, wind_dir, wind_speed, global_feat in train_loader:
            conc_map, wind_dir, wind_speed, global_feat = (
                conc_map.to(device),
                wind_dir.to(device),
                wind_speed.to(device),
                global_feat.to(device),
            )
            optimizer.zero_grad()
            output = model(
                conc_map,
                torch.stack([wind_speed, wind_dir], dim=1),
                global_features=global_feat,
            )
            wind_vec = (
                torch.cos(torch.deg2rad(wind_dir)).mean().item(),
                torch.sin(torch.deg2rad(wind_dir)).mean().item(),
            )
            loss, comps = physics_masked_loss_piml(output, conc_map, binary_map, wind_vector=wind_vec)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for conc_map, wind_dir, wind_speed, global_feat in val_loader:
                conc_map, wind_dir, wind_speed, global_feat = (
                    conc_map.to(device),
                    wind_dir.to(device),
                    wind_speed.to(device),
                    global_feat.to(device),
                )
                output = model(
                    conc_map,
                    torch.stack([wind_speed, wind_dir], dim=1),
                    global_features=global_feat,
                )
                wind_vec = (
                    torch.cos(torch.deg2rad(wind_dir)).mean().item(),
                    torch.sin(torch.deg2rad(wind_dir)).mean().item(),
                )
                v_loss, comps = physics_masked_loss_piml(output, conc_map, binary_map, wind_vector=wind_vec)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"[RetrainService] Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[RetrainService] 🔥 Nuovo best model salvato (val_loss={best_val_loss:.6f})")

    # === ⏱️ FINE TEMPO DI TRAINING ===
    duration_min = round((time() - t_start) / 60, 2)
    print(f"[RetrainService] ⏱️ Durata totale retraining: {duration_min} minuti")

    # --- Aggiornamento registry ---
    os.makedirs(LOGS_DIR, exist_ok=True)
    new_version = f"PIML_v{datetime.utcnow().strftime('%H%M%S')}"
    metrics = {
        "final_val_loss": round(best_val_loss, 6),
        "samples": len(dataset),
        "epochs": epochs,
        "duration_min": duration_min   # ✅ aggiunto qui
    }

    return new_version, metrics

# ============================================================
# TEST MANUALE (locale nel container)
# ============================================================

if __name__ == "__main__":
    retrain_model()
