import os

# Cartelle da ignorare
EXCLUDE_DIRS = {'.venv', '__pycache__', '.git', 'node_modules', '.idea', '.pytest_cache'}

def get_structure(root_dir, prefix=""):
    lines = []
    items = sorted(os.listdir(root_dir))
    for idx, item in enumerate(items):
        path = os.path.join(root_dir, item)
        if item in EXCLUDE_DIRS:
            continue
        connector = "└── " if idx == len(items) - 1 else "├── "
        lines.append(prefix + connector + item)
        if os.path.isdir(path):
            extension = "    " if idx == len(items) - 1 else "│   "
            lines.extend(get_structure(path, prefix + extension))
    return lines

if __name__ == "__main__":
    base_dir = "."  # cambia se vuoi partire da un'altra cartella
    output_file = "project_structure.txt"

    structure = get_structure(base_dir)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("📁 Project Structure\n")
        f.write("\n".join(structure))

    print(f"[INFO] Struttura salvata in {output_file}")
