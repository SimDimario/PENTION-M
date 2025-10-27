from MCxM import MCxM_CNN
import torch
import os
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from windrose import WindroseAxes
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gaussianPuff')))
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from CNNDataset import CNNDataset2
from train_model import train

def plot_plan_view(C1, x, y, title, wind_dir=None, wind_speed=None, puff_list=None, stability_class=1, n_show=10):
    fig, ax_main = plt.subplots(figsize=(8, 6))

    # Integra la concentrazione nel tempo lungo l'asse 2 (T)
    #data = np.trapz(C1, axis=2) * 1e6  # µg/m³ #type:ignore

    if isinstance(C1, torch.Tensor):
        C1 = C1.detach().cpu().numpy()
    
    data=C1
    vmin = np.percentile(data, 5)
    vmax = np.percentile(data, 95)

    # Plot della concentrazione integrata
    pcm = ax_main.pcolor(x, y, data, cmap='jet', shading='auto', vmin=vmin, vmax=vmax) #type:ignore
    fig.colorbar(pcm, ax=ax_main, label=r'$\mu g \cdot m^{-3}$')
    ax_main.set_xlabel('x (m)')
    ax_main.set_ylabel('y (m)')
    ax_main.set_title(title)
    ax_main.axis('equal')

    if wind_dir is not None and wind_speed is not None:
        inset_pos = [0.65, 0.65, 0.3, 0.3]  # left, bottom, width, height in figure coords
        ax_inset = WindroseAxes(fig, inset_pos)
        fig.add_axes(ax_inset)

        # Plot rosa dei venti con direzioni e velocità
        wind_dir = np.array(wind_dir) % 360
        wind_speed = np.full_like(wind_dir, fill_value=wind_speed, dtype=float)
        ax_inset.bar(wind_dir, wind_speed, normed=True, opening=0.8, edgecolor='white')
        ax_inset.set_legend(loc='lower right', title='Wind speed (m/s)')
        ax_inset.set_title("Rosa dei venti")

    # Plot puff sopra la plan view
    if puff_list is not None and len(puff_list) > 0:
        # Parametri σ_y empirici per classi A-F (Pasquill-Gifford)
        a_vals = [0.22, 0.16, 0.11, 0.08, 0.06, 0.04]
        b_vals = [0.90, 0.88, 0.86, 0.83, 0.80, 0.78]
        a = a_vals[stability_class - 1]
        b = b_vals[stability_class - 1]

        for i, puff in enumerate(puff_list):
            # if i % n_show != 0:
            #     continue  # salta puff intermedi

            distance = np.sqrt(puff.x**2 + puff.y**2)
            sigma_y = a * (distance + 1)**b  # evita 0^b

            circle = Circle((puff.x, puff.y), 2 * sigma_y, color='white', fill=False, lw=1.5)
            ax_main.add_patch(circle)
            ax_main.plot(puff.x, puff.y, 'wo', markersize=3)

        ax_main.legend(["Puff center (2σ)"], loc='lower right')

    plt.tight_layout()
    plt.show()

def smooth_curve(values, window=3):
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    pad_left = [values[0]] * (window//2)
    pad_right = [values[-1]] * (window - 1 - window//2)
    return np.concatenate([pad_left, smoothed, pad_right])

def plot_training_curves(train_losses, val_losses, val_maes, smooth_window=3):
    epochs = range(1, len(train_losses)+1)

    train_s = smooth_curve(train_losses, smooth_window)
    val_s = smooth_curve(val_losses, smooth_window)
    mae_s = smooth_curve(val_maes, smooth_window)

    plt.figure(figsize=(9,6))
    plt.plot(epochs, train_s, label="Train Loss", color="blue")
    plt.plot(epochs, val_s, label="Val Loss", color="orange")
    plt.plot(epochs, mae_s, label="Val MAE", color="green")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training & Validation Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

def rotate_map(cm, k):
    """Ruota la mappa di concentrazione di 90° k volte"""
    return np.rot90(cm, k)

def normalize_free_pixels(cm, mask):
        """
        cm: np.ndarray (m, m)
        mask: np.ndarray binaria (m, m) -> 1 = libero, 0 = edificio
        """
        free_vals = cm[mask==1]
        vmin, vmax = free_vals.min(), free_vals.max()
        cm_norm = cm.copy()
        cm_norm[mask==1] = (free_vals - vmin) / (vmax - vmin + 1e-8)
        return cm_norm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_MAP_PATH = os.path.join(os.path.dirname(__file__), "binary_maps_data/roma_italy_bbox.npy")
METADATA_MAP_PATH = os.path.join(SCRIPT_DIR, "binary_maps_data", "roma_italy_metadata_bbox.json")
REAL_CONC_PATH = os.path.join(SCRIPT_DIR, "dataset", "real_dispersion")
CSV_PATH = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-09-08_processed.csv")

if __name__ == "__main__":
    
    binary_map = np.load(BINARY_MAP_PATH)
    print(binary_map.shape)

    csv_df = pd.read_csv(CSV_PATH)
    csv_df_reduced = csv_df.groupby('simulation_id').first().reset_index()
    csv_df_reduced = csv_df_reduced[['wind_dir_cos', 'wind_dir_sin','wind_speed']]

    print(type(csv_df_reduced["wind_dir_cos"][0]))
    print(type(csv_df_reduced["wind_dir_sin"][0]))
    print(type(csv_df_reduced["wind_speed"][0]))
    
    m = 500

    with open(os.path.join(METADATA_MAP_PATH), 'r') as f:
        metadata = json.load(f)

    concentration_maps = []
    wind_dirs = []
    wind_speeds = []

    for file in tqdm(os.listdir(REAL_CONC_PATH), desc="Loading concentration maps"):
        conc_map = np.load(os.path.join(REAL_CONC_PATH, file))
        conc_map_mean = np.mean(conc_map, axis=2)  # Assuming conc_map is of shape (m, m, n)
        i = int(file.split('_')[1])  # file name format is 'sim_i_conc_real_...'
        wind_dir_cos, wind_dir_sin, wind_speed = csv_df_reduced.iloc[i]
        concentration_maps.append(conc_map_mean)

        rad_angle = np.arctan2(wind_dir_sin, wind_dir_cos)
        degree_angle = np.degrees(rad_angle)

        wind_dirs.append(degree_angle)
        wind_speeds.append(wind_speed)

    #plot_plan_view(concentration_maps[34], np.arange(m), np.arange(m), "Sample Processed Concentration Map", wind_dir=wind_dirs, wind_speed=wind_speeds[0], puff_list=None, stability_class=1, n_show=10)

    augmented_concentration_maps = []
    augmented_wind_dirs = []
    augmented_wind_speeds = []

    # --- Data augmentation with rotations ---
    print("[INFO] Performing data augmentation with rotations.")
    for cm, wd, ws in zip(concentration_maps, wind_dirs, wind_speeds):
        for k in range(4):  # 0°, 90°, 180°, 270°
            rotated_cm = rotate_map(cm, k)
            
            # Ruota anche il vento di conseguenza
            new_wd = (wd + k*90) % 360
            
            augmented_concentration_maps.append(rotated_cm)
            augmented_wind_dirs.append(new_wd)
            augmented_wind_speeds.append(ws)

    # --- Normalize concentration maps globally ---
    """all_values = np.concatenate([cm.flatten() for cm in augmented_concentration_maps])
    vmin, vmax = np.min(all_values), np.max(all_values)
    print(f"Global min: {vmin}, max: {vmax}")

    augmented_concentration_maps = [(cm - vmin) / (vmax - vmin) for cm in augmented_concentration_maps]
    print(augmented_concentration_maps[0].shape)"""

    #plot_plan_view(concentration_maps[34], np.arange(m), np.arange(m), "Sample Concentration Map", wind_dir=wind_dirs, wind_speed=wind_speeds[0], puff_list=None, stability_class=1, n_show=10)
    #plot_plan_view(binary_map*concentration_maps[34], np.arange(m), np.arange(m), "Binary Building Map", wind_dir=None, wind_speed=None, puff_list=None, stability_class=1, n_show=10)
    
    augmented_concentration_maps = [
        normalize_free_pixels(cm, binary_map) for cm in augmented_concentration_maps
    ]

    # ---  gloal features for binary map---
    city_features = np.array([
        metadata.get('building_density', 0.0),
        metadata.get('mean_height', 0.0),
        metadata.get('total_buildings', 0.0),
        metadata.get('free_cells', 0.0),
        metadata.get('total_cells', 0.0),
    ], dtype=np.float32)

    print("City features:", city_features)

    global_features = np.tile(city_features, (len(augmented_concentration_maps), 1))  # shape: [num_samples, 3]

    print("[INFO] Initializing CNNDataset.")
    dataset = CNNDataset2(augmented_concentration_maps, augmented_wind_dirs, augmented_wind_speeds, global_features=None, m=m)

    dataset_size = len(dataset)
    val_size = int(0.4 * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # type: ignore
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # type: ignore
        
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[INFO] Using device: {device}")

    # Initialize the model
    epochs=10
    model = MCxM_CNN(binary_map, m=m, n_channel=1, n_mask_correction=3, wind_dim=2, n_global_features=0).to(device)
    
    # Main training loop
    print("[INFO] Starting training loop.")
    model, output, train_losses, val_losses, val_maes = train(epochs, model, train_loader, val_loader, binary_map, device)

    print("[INFO] Training complete.")
    plot_training_curves(train_losses, val_losses, val_maes, smooth_window=3)

    # Plot output for a sample
    sample_map = output[5].detach().cpu().numpy()  # shape: (m, m)  # type:ignore
    plot_plan_view(sample_map, np.arange(m), np.arange(m), "Sample Output Map")

    # Save the model
    model_path = os.path.join(SCRIPT_DIR, "models", "mcxm_cnn_model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved to {model_path}")
