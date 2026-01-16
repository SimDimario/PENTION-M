import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(BASE_DIR, "datasetNPS", "PENTION_EI_Complete.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_model")
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_nps_model.json")
SCALER_PATH = os.path.join(MODEL_DIR, "xgb_scaler.pkl")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)


def augment_spectrum(s):
    s = s.copy().astype(float)

    shift = np.random.randint(-1, 2)
    if shift != 0:
        s = np.roll(s, shift)

    drift = np.linspace(
        np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), len(s)
    )
    s = s + drift
    s = s * (1 + np.random.normal(0, 0.03, len(s)))

    dropout = np.random.rand(len(s)) < 0.02
    s[dropout] = 0
    s = np.clip(s, 0, None)
    s = s ** np.random.uniform(0.92, 1.05)
    s = np.clip(s, 0, 100)
    return s


print("Preparing features...")

if "Name" in df.columns:
    df = df.drop(columns=["Name"])

X = df.drop(columns=["label"]).values
y = df["label"].values

X_aug = []
y_aug = []

for spectrum, lab in zip(X, y):
    for _ in range(5):
        X_aug.append(augment_spectrum(spectrum))
        y_aug.append(lab)

X = np.vstack([X] + X_aug)
y = np.hstack([y] + y_aug)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")
print("Training XGBoost...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=7,
    eval_metric="mlogloss",
    tree_method="hist",
    learning_rate=0.05,
    n_estimators=600,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)

model.fit(X_train_scaled, y_train)

model.save_model(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
print("Evaluating model...")

y_pred = model.predict(X_test_scaled)
report = classification_report(y_test, y_pred, output_dict=False)
print("\n=== CLASSIFICATION REPORT ===\n")
print(report)

with open(os.path.join(PLOTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print("Saving confusion matrix...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
plt.close()
print("Saving feature importance...")

importance = model.feature_importances_
plt.figure(figsize=(10, 5))
plt.plot(importance)
plt.title("Feature Importance - XGBoost")
plt.xlabel("m/z index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"))
plt.close()
print("\nDONE!")
print(f"All plots saved inside: {PLOTS_DIR}")
