import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# === CONFIG ===
old_dataset_path = "ClassificatoreNPS/datasetNPS/1-s2.0-S2468170923000358-mmc1.csv"
new_dataset_path = "ClassificatoreNPS/datasetNPS/swgdrug_spectra_600.csv"
model_path = "ClassificatoreNPS/model/dnn_spectra_version.keras"
scaler_path = "ClassificatoreNPS/model/scale_dnn.pkl"
output_path = "ClassificatoreNPS/datasetNPS/swgdrug_labeled.csv"
confidence_output = "ClassificatoreNPS/datasetNPS/prediction_confidence.csv"
confidence_threshold = 0.8

# === CARICAMENTO ===
print("[INFO] Caricamento dati e modello DNN...")
old_df = pd.read_csv(old_dataset_path)
new_df = pd.read_csv(new_dataset_path)
model = keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# === MATCHING DIRETTO PER NAME ===
print("[INFO] Etichettatura tramite matching diretto...")
name_to_label = dict(zip(old_df["Name"].str.strip().str.lower(), old_df["label"]))
new_df["match_key"] = new_df["Name"].str.strip().str.lower()
new_df["label"] = new_df["match_key"].map(name_to_label)

# === PREDIZIONE PER I NON MATCHATI ===
print("[INFO] Predizione automatica per i composti non etichettati...")
unlabeled_mask = new_df["label"].isna()
unlabeled_df = new_df[unlabeled_mask].copy()

spectrum_cols = [col for col in new_df.columns if col.isdigit()]
X_unlabeled = unlabeled_df[spectrum_cols].astype(float).values
X_unlabeled_scaled = scaler.transform(X_unlabeled)

# Predici probabilità
proba = model.predict(X_unlabeled_scaled, verbose=1)
pred_labels = np.argmax(proba, axis=1)
pred_confidence = np.max(proba, axis=1)
final_labels = np.where(pred_confidence >= confidence_threshold, pred_labels, 6)

# === SALVA CONFIDENZA DELLA PREVISIONE ===
confidence_df = unlabeled_df[["Name"]].copy()
confidence_df["PredictedLabel"] = pred_labels
confidence_df["Confidence"] = pred_confidence
confidence_df["FinalLabel"] = final_labels
confidence_df.to_csv(confidence_output, index=False)
print(f"[✓] File confidenze salvato in: {confidence_output}")

# === ASSEGNA LABEL ===
new_df.loc[unlabeled_mask, "label"] = final_labels
new_df["label"] = new_df["label"].astype(int)

print(f"\n[✓] Composti etichettati direttamente: {len(new_df) - len(unlabeled_df)}")
print(f"[✓] Composti etichettati dal DNN: {len(unlabeled_df)}")
print(f"[✓] Di cui classificati come 'Other': {(final_labels == 6).sum()}")

# === ESPORTAZIONE ===
new_df.drop(columns=["match_key"], inplace=True)
new_df.to_csv(output_path, index=False)
print(f"[✓] File finale etichettato salvato in: {output_path}")
