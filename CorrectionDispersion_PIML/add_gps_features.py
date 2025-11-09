import pandas as pd
import os

# Percorso dataset originale
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-11-09_PIML_processed.csv")

# Caricamento CSV
df = pd.read_csv(csv_path)

# Normalizzazione coordinate GPS (0–1)
df["gps_x"] = df["source_x"] / df["source_x"].max()
df["gps_y"] = df["source_y"] / df["source_y"].max()

# Salvataggio nuovo dataset
output_path = os.path.join(SCRIPT_DIR, "dataset", "nps_simulated_dataset_gaussiano_2025-11-09_PIML_GPS.csv")
df.to_csv(output_path, index=False)

print(f"[INFO] Nuovo dataset salvato in: {output_path}")
print(df[["simulation_id", "gps_x", "gps_y"]].head())
