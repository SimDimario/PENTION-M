import os

# Cartelle da ignorare completamente
EXCLUDE_DIRS = {'.venv', '__pycache__', '.git', 'node_modules', '.idea', '.pytest_cache', 'test_datasetNPS', '.vscode'}

# Estensioni di file da ignorare (temporanei LaTeX)
EXCLUDE_EXT = {
    ".aux", ".bbl", ".bcf", ".blg", ".fdb_latexmk", ".fls", ".lof",
    ".log", ".lot", ".run.xml", ".synctex.gz", ".toc", ".out"
}

# Cartelle da comprimere
COMPRESS_DIRS = {
    "logs/forensic",
    "CorrectionDispersion_PIML/dataset/real_dispersion"
}

def should_compress(path: str) -> bool:
    """True se il path è dentro una delle cartelle da comprimere."""
    norm_path = path.replace("\\", "/")
    for cd in COMPRESS_DIRS:
        if cd in norm_path:
            return True
    return False


def is_excluded_file(item: str) -> bool:
    """
    Ritorna True se il file ha un'estensione da ignorare.
    Gestisce estensioni composte come .synctex.gz
    """
    if "." not in item:
        return False
    
    # Estensione completa (es: main.synctex.gz)
    full_ext = "." + item.split(".", 1)[1]
    
    return full_ext in EXCLUDE_EXT


def list_with_compression(path):
    items = sorted(os.listdir(path))

    # Non comprimere se cartella piccola
    if not should_compress(path) or len(items) <= 10:
        return items

    # Compressione
    return items[:3] + ["..."] + items[-3:]


def get_structure(root_dir, prefix=""):
    lines = []
    items = list_with_compression(root_dir)

    for idx, item in enumerate(items):
        path = os.path.join(root_dir, item)

        # Escludi directory non desiderate
        if item in EXCLUDE_DIRS:
            continue

        # Escludi file indesiderati (.aux, .log, .toc ecc.)
        if os.path.isfile(path) and is_excluded_file(item):
            continue

        connector = "└── " if idx == len(items) - 1 else "├── "

        # Elemento placeholder "..."
        if item == "...":
            lines.append(prefix + "│   " + "...")
            continue

        lines.append(prefix + connector + item)

        # Ricorsione per cartelle
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
