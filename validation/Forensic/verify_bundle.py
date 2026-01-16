import json
import hashlib
import sys
import os
from glob import glob
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

def get_latest_bundle():
    bundle_dir = os.path.join("logs", "forensic")
    paths = sorted(glob(os.path.join(bundle_dir, "bundle_*.json")), key=os.path.getmtime)
    if not paths:
        print("No bundles found in logs/forensic/")
        sys.exit(1)
    return paths[-1]

if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg == "--latest":
        bundle_path = get_latest_bundle()
    else:
        bundle_path = arg
else:
    bundle_path = "bundle.json"

print(f"Checking: {bundle_path}")

if not os.path.exists(bundle_path):
    print("File not found")
    sys.exit(1)
with open(bundle_path, "r") as f:
    bundle = json.load(f)
event = bundle["event"]

canonical_event = json.loads(json.dumps(event, sort_keys=True, default=str))
serialized = json.dumps(canonical_event, sort_keys=True, default=str)
computed_hash = hashlib.sha256(serialized.encode()).hexdigest()
print("Calculated hash:", computed_hash)
print("Hash in the file:", bundle["hash_sha256"])
print()

if computed_hash != bundle["hash_sha256"]:
    print("HASH DOES NOT MATCH → FILE MODIFIED")
    hash_ok = False
else:
    print("HASH OK")
    hash_ok = True
print()
print("Verify digital signature...")
signature = bytes.fromhex(bundle["signature"])
pub_key_pem = bundle["public_key"].encode()
try:
    pub_key = serialization.load_pem_public_key(pub_key_pem)
    pub_key.verify(signature, computed_hash.encode())
    print("VALID DIGITAL SIGNATURE → AUTHENTIC BUNDLE")
    sig_ok = True
except InvalidSignature:
    print("INVALID SIGNATURE → TAMPERED FILE")
    sig_ok = False
except Exception as e:
    print("Error in signature verification:", e)
    sig_ok = False
print()

if hash_ok and sig_ok:
    print("COMPLETE Verification: The bundle is intact and authentic.")
else:
    print("The bundle is NOT intact or has been altered.")