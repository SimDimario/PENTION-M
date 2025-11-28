import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# === Path dinamici (nessuna data hardcoded) ===
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Cerca automaticamente l’ultimo dataset PIML disponibile
files = sorted(
    [f for f in os.listdir(DATASET_DIR) if f.startswith("nps_simulated_dataset_gaussiano") and f.endswith("_PIML.csv")],
    reverse=True
)
if not files:
    raise FileNotFoundError("❌ Nessun dataset PIML trovato in ./dataset/")
latest_file = files[0]

DATA_PATH = os.path.join(DATASET_DIR, latest_file)
OUTPUT_PATH = DATA_PATH.replace("_PIML.csv", "_PIML_processed.csv")

print(f"📂 Dataset trovato: {latest_file}")
print(f"📤 Output previsto: {os.path.basename(OUTPUT_PATH)}")

# === Carica dataset ===
dataset = pd.read_csv(DATA_PATH)
print(f"\nNumero righe: {dataset.shape[0]}, colonne: {dataset.shape[1]}")
print(dataset.head(5))
print("\nMissing values per colonna:\n", dataset.isnull().sum())
print("\nStatistiche di base:\n", dataset.describe())

# === Distribuzione variabili continue ===
continuous_cols = [c for c in ["wind_speed", "sigma_y", "sigma_z", "stability_index", "RH"] if c in dataset.columns]
if continuous_cols:
    dataset[continuous_cols].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Distribuzione variabili fisiche PIML", fontsize=16)
    plt.savefig(os.path.join(PLOT_DIR, "distribuzioni_variabili_continue.png"))
    plt.close()
else:
    print("⚠️ Nessuna colonna continua trovata per la distribuzione.")

# === Distribuzione variabili categoriche ===
categorical_cols = [c for c in ["stability_class", "dispersion_model"] if c in dataset.columns]
if categorical_cols:
    fig, axes = plt.subplots(1, len(categorical_cols), figsize=(10, 5))
    if len(categorical_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, categorical_cols):
        sns.countplot(data=dataset, x=col, ax=ax)
        ax.set_title(f"Distribuzione di {col}")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "distribuzioni_variabili_categoriche.png"))
    plt.close()

# === Calcolo componenti vento (cos/sin) da direzione media ===
if "wind_dir_mean" in dataset.columns:
    dataset["wind_dir_cos"] = np.cos(np.deg2rad(dataset["wind_dir_mean"]))
    dataset["wind_dir_sin"] = np.sin(np.deg2rad(dataset["wind_dir_mean"]))

# === Encoding per variabili categoriche ===
label_cols = [c for c in ["stability_class", "dispersion_model"] if c in dataset.columns]
if label_cols:
    encoders = {col: LabelEncoder().fit(dataset[col].astype(str)) for col in label_cols}
    for col, le in encoders.items():
        dataset[col] = le.transform(dataset[col].astype(str))

# === Matrice di correlazione ===
corr = dataset.corr(numeric_only=True)
plt.figure(figsize=(12, 7))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Matrice di correlazione variabili PIML")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "matrice_correlazione.png"))
plt.close()

# === Salva dataset processato ===
dataset.to_csv(OUTPUT_PATH, index=False)
print(f"\nDataset PIML processato e salvato in:\n{OUTPUT_PATH}")
print(f"Grafici salvati in: {PLOT_DIR}")
