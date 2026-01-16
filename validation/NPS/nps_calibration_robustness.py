import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_PATH = os.path.join(ROOT, "ClassificatoreNPS/datasetNPS/PENTION_EI_Complete.csv")
MODEL_PATH = os.path.join(ROOT, "ClassificatoreNPS/model/xgb_nps_model.json")
SCALER_PATH = os.path.join(ROOT, "ClassificatoreNPS/model/xgb_scaler.pkl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_nps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Loading XGBoost model...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print("[INFO] Loading scaler...")
scaler = joblib.load(SCALER_PATH)
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)
if "Name" in df.columns:
    df = df.drop(columns=["Name"])
feature_cols = [c for c in df.columns if c not in ["label"]]
X = df[feature_cols].values.astype(float)
y = df["label"].values
X_scaled = scaler.transform(X)
print("[INFO] Computing predictions...")

from scipy.special import softmax

def apply_temp_scaling(raw_probs, T=1.8):
    logits = np.log(np.clip(raw_probs, 1e-12, 1))
    logits_scaled = logits / T
    return softmax(logits_scaled, axis=1)

raw_probs = model.predict_proba(X_scaled)
probs = apply_temp_scaling(raw_probs, T=1.8)
max_conf = probs.max(axis=1)
preds = probs.argmax(axis=1)

plt.figure(figsize=(7,5))
plt.hist(max_conf, bins=30, edgecolor='black')
plt.title("Distribution of XGBoost Confidences (Temp Scaling T=1.8)")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.savefig(os.path.join(OUTPUT_DIR, "confidence_histogram.png"), dpi=200)
plt.close()
print("[INFO] Computing calibration curve...")

from sklearn.calibration import calibration_curve

correct = (preds == y).astype(int)
prob_true, prob_pred = calibration_curve(
    correct, max_conf, n_bins=10, strategy="quantil"
)

plt.figure(figsize=(7,7))
plt.plot(prob_pred, prob_true, "o-", label="XGB+TempScaling (T=1.8)")
plt.plot([0,1], [0,1], "--", label="Perfect calibration")
plt.xlabel("Predicted confidence")
plt.ylabel("True accuracy")
plt.title("Calibration Curve (with T=1.8)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, "calibration_curve.png"), dpi=200)
plt.close()
brier = brier_score_loss(correct, max_conf)
print("[INFO] Running noise robustness test...")

def add_noise(sample, noise_level):
    noisy = sample + np.random.normal(0, noise_level, size=sample.shape)
    noisy = np.clip(noisy, 0, None)
    return noisy

noise_levels = [0.01, 0.05, 0.1, 0.2]
noise_results = []

for nl in noise_levels:
    noisy_samples = np.array([add_noise(x, nl) for x in X])
    noisy_scaled = scaler.transform(noisy_samples)
    noisy_probs = model.predict_proba(noisy_scaled).max(axis=1)
    noise_results.append({
        "noise": nl,
        "mean_conf": float(noisy_probs.mean()),
        "std_conf": float(noisy_probs.std())
    })

noise_df = pd.DataFrame(noise_results)
noise_df.to_csv(os.path.join(OUTPUT_DIR, "noise_robustness.csv"), index=False)

plt.figure(figsize=(7,5))
plt.plot(noise_df["noise"], noise_df["mean_conf"], "o-", label="Mean confidence")
plt.xlabel("Noise level")
plt.ylabel("Mean confidence")
plt.title("Robustness to Spectral Noise")
plt.grid()
plt.savefig(os.path.join(OUTPUT_DIR, "noise_robustness.png"), dpi=200)
plt.close()

summary = {
    "brier_score": float(brier),
    "avg_confidence": float(max_conf.mean()),
    "std_confidence": float(max_conf.std())
}

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("[DONE] Results saved in:", OUTPUT_DIR)