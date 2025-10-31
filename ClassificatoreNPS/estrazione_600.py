import csv
import re

def parse_msp_fixed_600(filepath, max_mz=600):
    compounds = []
    current = {}
    peaks = [0.0] * max_mz

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.strip()

            # blocco di fine composto (riga vuota)
            if not line:
                if current:
                    current["spectrum"] = peaks.copy()
                    compounds.append(current.copy())
                current = {}
                peaks = [0.0] * max_mz
                continue

            # parsing nome composto
            if line.lower().startswith("name:"):
                current["name"] = line.split(":", 1)[1].strip()
                continue

            # parsing righe con picchi multipli (es: "40 42; 41 493; 42 63; ...")
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
                    except Exception:
                        continue

    return compounds

def save_spectra_csv(compounds, output_csv, max_mz=600):
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["Name"] + [f"m/z_{i}" for i in range(max_mz)]
        writer.writerow(header)

        for c in compounds:
            row = [c.get("name", "")] + c.get("spectrum", [0.0]*max_mz)
            writer.writerow(row)

# ====== USO ======
msp_file = "ClassificatoreNPS/datasetNPS/SWGDRUG 3.MSP"           
output_csv = "ClassificatoreNPS/datasetNPS/swgdrug_spectra_600.csv"  

compounds = parse_msp_fixed_600(msp_file)
save_spectra_csv(compounds, output_csv)

print(f"[✓] Esportati {len(compounds)} composti con 600 colonne m/z in: {output_csv}")
