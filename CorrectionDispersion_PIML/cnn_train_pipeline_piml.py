import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from windrose import WindroseAxes

# === FIX PATHS ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "CorrectionDispersion_PIML"))

# === IMPORT PIML MODULES ===
from MCxM_PIML import MCxM_PIML
from loss_function_piml import physics_masked_loss_piml
from CNNDataset import CNNDataset2

# === UTILS ===
def rotate_map(cm, k): return np.rot90(cm, k)
def normalize_free_pixels(cm, mask):
    free_vals = cm[mask == 1]
    vmin, vmax = free_vals.min(), free_vals.max()
    cm_norm = cm.copy()
    cm_norm[mask == 1] = (free_vals - vmin) / (vmax - vmin + 1e-8)
    return cm_norm

def smooth_curve(values, window=3):
    if len(values) < window: return values
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    pad_left = [values[0]] * (window//2)
    pad_right = [values[-1]] * (window - 1 - window//2)
    return np.concatenate([pad_left, smoothed, pad_right])

def plot_training_curves(train_losses, val_losses, save_path=None):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(9,6))
    plt.plot(epochs, smooth_curve(train_losses), label="Train Loss", color="blue")
    plt.plot(epochs, smooth_curve(val_losses), label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss (PIML)")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# === PATHS ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data/amsterdam_netherlands_bbox.npy")
METADATA_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data/amsterdam_netherlands_metadata_bbox.json")
REAL_CONC_PATH = os.path.join(SCRIPT_DIR, "dataset", "real_dispersion")
CSV_PATH = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-11-10_PIML_processed.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# === TRAINING ===
if __name__ == "__main__":
    binary_map = np.load(BINARY_MAP_PATH)
    m = binary_map.shape[0]

    # === Caricamento dataset ===
    csv_df = pd.read_csv(CSV_PATH)
    csv_df_reduced = csv_df.groupby('simulation_id').first().reset_index()
    csv_df_reduced = csv_df_reduced[['wind_dir_cos', 'wind_dir_sin', 'wind_speed', 'gps_x', 'gps_y']]

    with open(METADATA_MAP_PATH, 'r') as f:
        metadata = json.load(f)

    concentration_maps, wind_dirs, wind_speeds = [], [], []
    for file in tqdm(os.listdir(REAL_CONC_PATH), desc="Loading maps"):
        conc_map = np.load(os.path.join(REAL_CONC_PATH, file))
        conc_map_mean = np.mean(conc_map, axis=2)
        i = int(file.split('_')[1])
        wind_dir_cos, wind_dir_sin, wind_speed, gps_x, gps_y = csv_df_reduced.iloc[i]
        rad_angle = np.arctan2(wind_dir_sin, wind_dir_cos)
        degree_angle = np.degrees(rad_angle)
        concentration_maps.append(conc_map_mean)
        wind_dirs.append(degree_angle)
        wind_speeds.append(wind_speed)

    gps_features = list(zip(csv_df_reduced["gps_x"], csv_df_reduced["gps_y"]))

    # --- Augmentation e normalizzazione (include GPS) ---
    aug_maps, aug_dirs, aug_speeds, aug_gps = [], [], [], []
    for (cm, wd, ws, gps) in zip(concentration_maps, wind_dirs, wind_speeds, gps_features):
        for k in range(4):
            aug_maps.append(rotate_map(cm, k))
            aug_dirs.append((wd + k*90) % 360)
            aug_speeds.append(ws)
            aug_gps.append(gps)  # stesso GPS per tutte le rotazioni
    aug_maps = [normalize_free_pixels(cm, binary_map) for cm in aug_maps]

    dataset = CNNDataset2(aug_maps, aug_dirs, aug_speeds, global_features=aug_gps, m=m)
    train_dataset, val_dataset = random_split(dataset, [int(0.6*len(dataset)), int(0.4*len(dataset))])
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    # === GPU / DEVICE ===
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"[INFO] Using device: {device}")

    # === MODELLO PIML ===
    model = MCxM_PIML(binary_map, m=m, n_channel=1, wind_dim=2, n_global_features=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # === PARAMETRI TRAINING ===
    epochs = 80
    patience = 10  # early stopping
    best_val_loss = np.inf
    patience_counter = 0
    train_losses, val_losses = [], []

    # === TRAIN LOOP ===
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for conc_map, wind_dir, wind_speed, global_feat in train_loader:
            conc_map, wind_dir, wind_speed, global_feat = (
                conc_map.to(device),
                wind_dir.to(device),
                wind_speed.to(device),
                global_feat.to(device)
            )
            optimizer.zero_grad()
            output = model(
                conc_map,
                torch.stack([wind_speed, wind_dir], dim=1),
                global_features=global_feat
            )

            wind_vec = (torch.cos(torch.deg2rad(wind_dir)).mean().item(),
                        torch.sin(torch.deg2rad(wind_dir)).mean().item())
            loss, comps = physics_masked_loss_piml(output, conc_map, binary_map, wind_vector=wind_vec)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === VALIDATION ===
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for conc_map, wind_dir, wind_speed, global_feat in val_loader:
                conc_map, wind_dir, wind_speed, global_feat = (
                    conc_map.to(device),
                    wind_dir.to(device),
                    wind_speed.to(device),
                    global_feat.to(device)
                )
                output = model(
                    conc_map,
                    torch.stack([wind_speed, wind_dir], dim=1),
                    global_features=global_feat
                )
                wind_vec = (torch.cos(torch.deg2rad(wind_dir)).mean().item(),
                            torch.sin(torch.deg2rad(wind_dir)).mean().item())
                val_loss, comps = physics_masked_loss_piml(output, conc_map, binary_map, wind_vector=wind_vec)
                val_running += val_loss.item()

        avg_val_loss = val_running / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"[EPOCH {epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # === EARLY STOPPING ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "models", "mcxm_piml_model_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                break

    # === Plot curve ===
    plot_training_curves(train_losses, val_losses,
                         save_path=os.path.join(PLOT_DIR, "training_curves_piml.png"))

    # === Salva modello finale ===
    model_path = os.path.join(SCRIPT_DIR, "models", "mcxm_piml_model_final.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Training completed. Model saved to {model_path}")
