import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

files = sorted(
    [
        f
        for f in os.listdir(DATASET_DIR)
        if f.startswith("nps_simulated_dataset_gaussiano") and f.endswith("_PIML.csv")
    ],
    reverse=True,
)
if not files:
    raise FileNotFoundError("No PIML datasets found in ./dataset/")
latest_file = files[0]

DATA_PATH = os.path.join(DATASET_DIR, latest_file)
OUTPUT_PATH = DATA_PATH.replace("_PIML.csv", "_PIML_processed.csv")

print(f"Dataset found: {latest_file}")
print(f"Expected output: {os.path.basename(OUTPUT_PATH)}")

dataset = pd.read_csv(DATA_PATH)
print(f"\nNumber of lines: {dataset.shape[0]}, columns: {dataset.shape[1]}")
print(dataset.head(5))
print("\nMissing values ​​per column:\n", dataset.isnull().sum())
print("\nBasic statistics:\n", dataset.describe())

continuous_cols = [
    c
    for c in ["wind_speed", "sigma_y", "sigma_z", "stability_index", "RH"]
    if c in dataset.columns
]
if continuous_cols:
    dataset[continuous_cols].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Distribution of PIML physical variables", fontsize=16)
    plt.savefig(os.path.join(PLOT_DIR, "distribuzioni_variabili_continue.png"))
    plt.close()
else:
    print("No continuous columns found for distribution.")

categorical_cols = [
    c for c in ["stability_class", "dispersion_model"] if c in dataset.columns
]
if categorical_cols:
    fig, axes = plt.subplots(1, len(categorical_cols), figsize=(10, 5))
    if len(categorical_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, categorical_cols):
        sns.countplot(data=dataset, x=col, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "distribuzioni_variabili_categoriche.png"))
    plt.close()

if "wind_dir_mean" in dataset.columns:
    dataset["wind_dir_cos"] = np.cos(np.deg2rad(dataset["wind_dir_mean"]))
    dataset["wind_dir_sin"] = np.sin(np.deg2rad(dataset["wind_dir_mean"]))

label_cols = [
    c for c in ["stability_class", "dispersion_model"] if c in dataset.columns
]
if label_cols:
    encoders = {col: LabelEncoder().fit(dataset[col].astype(str)) for col in label_cols}
    for col, le in encoders.items():
        dataset[col] = le.transform(dataset[col].astype(str))

corr = dataset.corr(numeric_only=True)
plt.figure(figsize=(12, 7))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("PIML Variable Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "matrice_correlazione.png"))
plt.close()

dataset.to_csv(OUTPUT_PATH, index=False)
print(f"\nPIML dataset processed and saved to:\n{OUTPUT_PATH}")
print(f"Plots saved in: {PLOT_DIR}")
