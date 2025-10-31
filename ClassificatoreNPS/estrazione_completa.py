import csv
import re

def parse_msp_extended_fixed(filepath, max_mz=1200):
    compounds = []
    current = {
        "name": "", "formula": "", "mw": "", "exactmass": "",
        "inchikey": "", "casno": "", "comment": "", "id": ""
    }
    peaks = [0.0] * max_mz

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                if any(current.values()) or any(peaks):
                    current["spectrum"] = peaks.copy()
                    compounds.append(current.copy())
                current = {
                    "name": "", "formula": "", "mw": "", "exactmass": "",
                    "inchikey": "", "casno": "", "comment": "", "id": ""
                }
                peaks = [0.0] * max_mz
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in current:
                    current[key] = value
                continue

            if any(char.isdigit() for char in line):
                line = line.replace(";", " ")
                tokens = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                for i in range(0, len(tokens) - 1, 2):
                    try:
                        mz = float(tokens[i])
                        intensity = float(tokens[i + 1])
                        mz_index = int(round(mz))
                        if 0 <= mz_index < max_mz:
                            peaks[mz_index] = intensity
                    except:
                        continue

    return compounds

def save_extended_csv(compounds, output_csv, max_mz=1200):
    # Trova il massimo indice m/z effettivamente usato
    last_used_index = max_mz - 1
    while last_used_index >= 0:
        if any(c["spectrum"][last_used_index] != 0.0 for c in compounds):
            break
        last_used_index -= 1

    effective_mz = last_used_index + 1  # Per includere l'ultimo valido
    print(f"[INFO] Ultimo m/z utile: m/z_{last_used_index} → Salvo fino a {effective_mz} colonne")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = [
            "Name", "Formula", "MW", "ExactMass", "InChIKey",
            "CASNO", "Comment", "ID"
        ] + [str(i + 1) for i in range(max_mz)]
        writer.writerow(header)

        for c in compounds:
            row = [
                c.get("name", ""), c.get("formula", ""), c.get("mw", ""),
                c.get("exactmass", ""), c.get("inchikey", ""), c.get("casno", ""),
                c.get("comment", ""), c.get("id", "")
            ] + c["spectrum"][:effective_mz]
            writer.writerow(row)

# ====== USO ======
msp_file = "ClassificatoreNPS/datasetNPS/SWGDRUG 3.MSP"
output_csv = "ClassificatoreNPS/datasetNPS/swgdrug_full.csv"
mz_limit = 1200

compounds = parse_msp_extended_fixed(msp_file, max_mz=mz_limit)
save_extended_csv(compounds, output_csv, max_mz=mz_limit)

print(f"[✓] Esportati {len(compounds)} composti in {output_csv} con taglio automatico delle colonne inutilizzate.")
