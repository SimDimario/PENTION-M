import os

# Cartelle da ignorare completamente
EXCLUDE_DIRS = {'.venv', '__pycache__', '.git', 'node_modules', '.idea', '.pytest_cache', 'test_datasetNPS'}

# Cartelle da comprimere (metti esattamente quelle che vuoi)
COMPRESS_DIRS = {
    "logs/forensic",
    "CorrectionDispersion_PIML/dataset/real_dispersion"
}

def should_compress(path: str) -> bool:
    """
    Ritorna True se il path è dentro una delle cartelle da comprimere.
    """
    norm_path = path.replace("\\", "/")
    for cd in COMPRESS_DIRS:
        if cd in norm_path:
            return True
    return False


def list_with_compression(path):
    """
    Se la cartella è tra quelle da comprimere:
        - restituisce primi 3 file, '...', ultimi 3
    Altrimenti, restituisce tutti gli elementi normalmente.
    """
    items = sorted(os.listdir(path))

    # Non comprimere se cartella piccola
    if not should_compress(path) or len(items) <= 10:
        return items

    # Compressione SOLO delle cartelle elencate
    return items[:3] + ["..."] + items[-3:]


def get_structure(root_dir, prefix=""):
    lines = []
    items = list_with_compression(root_dir)

    for idx, item in enumerate(items):
        if item in EXCLUDE_DIRS:
            continue

        connector = "└── " if idx == len(items) - 1 else "├── "
        path = os.path.join(root_dir, item)

        # Se è l'elemento placeholder "..."
        if item == "...":
            lines.append(prefix + "│   " + "...")
            continue

        lines.append(prefix + connector + item)

        # Se è directory → ricorsione
        if os.path.isdir(path):
            extension = "    " if idx == len(items) - 1 else "│   "
            lines.extend(get_structure(path, prefix + extension))

    return lines


if __name__ == "__main__":
    base_dir = "."
    output_file = "project_structure.txt"

    structure = get_structure(base_dir)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("📁 Project Structure\n")
        f.write("\n".join(structure))

    print(f"[INFO] Struttura salvata in {output_file}")
