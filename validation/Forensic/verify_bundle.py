import json
import hashlib
import sys
import os
from glob import glob
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

# ------------------------------------------
#  Trova ultimo bundle generato
# ------------------------------------------
def get_latest_bundle():
    bundle_dir = os.path.join("logs", "forensic")
    paths = sorted(glob(os.path.join(bundle_dir, "bundle_*.json")), key=os.path.getmtime)
    if not paths:
        print("❌ Nessun bundle trovato in logs/forensic/")
        sys.exit(1)
    return paths[-1]

# ------------------------------------------
#  Determina file da verificare
# ------------------------------------------
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg == "--latest":
        bundle_path = get_latest_bundle()
    else:
        bundle_path = arg
else:
    bundle_path = "bundle.json"

print(f"📄 Verificando: {bundle_path}")

if not os.path.exists(bundle_path):
    print("❌ File non trovato")
    sys.exit(1)

# ------------------------------------------
#  Carica bundle
# ------------------------------------------
with open(bundle_path, "r") as f:
    bundle = json.load(f)

event = bundle["event"]

# ------------------------------------------
#  Ricostruisci canonical_event
# ------------------------------------------
canonical_event = json.loads(json.dumps(event, sort_keys=True, default=str))

serialized = json.dumps(canonical_event, sort_keys=True, default=str)
computed_hash = hashlib.sha256(serialized.encode()).hexdigest()

print("Hash calcolato :", computed_hash)
print("Hash nel file :", bundle["hash_sha256"])
print()

# ------------------------------------------
#  Check hash
# ------------------------------------------
if computed_hash != bundle["hash_sha256"]:
    print("❌ HASH NON CORRISPONDE → FILE MODIFICATO")
    hash_ok = False
else:
    print("✔ HASH OK")
    hash_ok = True

print()

# ------------------------------------------
#  Verifica firma digitale Ed25519
# ------------------------------------------
print("Verifica firma digitale...")

signature = bytes.fromhex(bundle["signature"])
pub_key_pem = bundle["public_key"].encode()

try:
    pub_key = serialization.load_pem_public_key(pub_key_pem)
    pub_key.verify(signature, computed_hash.encode())
    print("✔ FIRMA DIGITALE VALIDA → BUNDLE AUTENTICO")
    sig_ok = True
except InvalidSignature:
    print("❌ FIRMA NON VALIDA → FILE MANOMESSO")
    sig_ok = False
except Exception as e:
    print("❌ Errore nella verifica firma:", e)
    sig_ok = False

print()

if hash_ok and sig_ok:
    print("🎉 Verifica COMPLETA: il bundle è integro e autentico.")
else:
    print("⚠️ Il bundle NON è integro o è stato alterato.")
