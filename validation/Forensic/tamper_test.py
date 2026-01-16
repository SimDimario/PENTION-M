import json
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from validation.Forensic.forensic_validation import validate_bundle, BUNDLE_DIR

TAMPER = os.path.join(os.path.dirname(__file__), "tampered_bundle.json")


def create_tampered_copy():
    bundles = sorted([f for f in os.listdir(BUNDLE_DIR) if f.endswith(".json")])
    src = os.path.join(BUNDLE_DIR, bundles[-1])
    shutil.copy(src, TAMPER)
    with open(TAMPER, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["event"]["SensorGPS"]["latitude"] += 0.0001
    with open(TAMPER, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return TAMPER


if __name__ == "__main__":
    tampered_path = create_tampered_copy()
    print("\n=== VALIDATING TAMPERED BUNDLE ===")
    validate_bundle(tampered_path)
