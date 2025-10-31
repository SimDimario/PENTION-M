import pandas as pd

# === Percorsi file ===
old_path = "ClassificatoreNPS/datasetNPS/1-s2.0-S2468170923000358-mmc1.csv"
new_path = "ClassificatoreNPS/datasetNPS/swgdrug_labeled.csv"
output_path = "ClassificatoreNPS/datasetNPS/swgdrug_extended.csv"

# === Caricamento ===
print("[INFO] Caricamento dei dataset...")
old_df = pd.read_csv(old_path)
new_df = pd.read_csv(new_path)

# === Normalizza nomi per matching ===
old_df["Name_key"] = old_df["Name"].str.strip().str.lower()
new_df["Name_key"] = new_df["Name"].str.strip().str.lower()

# === Seleziona colonne comuni ===
common_cols = [col for col in old_df.columns if col in new_df.columns]

# === Filtra righe presenti solo nel vecchio ===
merged = old_df.merge(new_df, how="left", on="Name_key", indicator=True)
only_old = merged[merged["_merge"] == "left_only"]

# Prendi le righe originali dal vecchio (senza colonne duplicate e _merge)
missing_from_new = old_df[old_df["Name_key"].isin(only_old["Name_key"])].copy()
missing_from_new.drop(columns=["Name_key"], inplace=True)

print(f"[INFO] Composti unici trovati nel vecchio dataset: {len(missing_from_new)}")

# === Unione ===
extended_df = pd.concat([new_df.drop(columns=["Name_key"]), missing_from_new], ignore_index=True)

# === Salvataggio ===
extended_df.to_csv(output_path, index=False)
print(f"[✓] Dataset esteso salvato in: {output_path}")
