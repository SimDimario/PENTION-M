import os

EXCLUDE_DIRS = {'.venv', '__pycache__', '.git', 'node_modules', '.idea', '.pytest_cache', 'test_datasetNPS', '.vscode'}
EXCLUDE_EXT = {
    ".aux", ".bbl", ".bcf", ".blg", ".fdb_latexmk", ".fls", ".lof",
    ".log", ".lot", ".run.xml", ".synctex.gz", ".toc", ".out"
}
COMPRESS_DIRS = {
    "logs/forensic",
    "CorrectionDispersion_PIML/dataset/real_dispersion"
}

def should_compress(path: str) -> bool:
    """True if the path is within one of the folders to be compressed."""
    norm_path = path.replace("\\", "/")
    for cd in COMPRESS_DIRS:
        if cd in norm_path:
            return True
    return False

def is_excluded_file(item: str) -> bool:
    """
    Returns True if the file has an extension to be ignored.
    Handles compound extensions like .synctex.gz
    """
    if "." not in item:
        return False
    
    full_ext = "." + item.split(".", 1)[1]
    return full_ext in EXCLUDE_EXT

def list_with_compression(path):
    items = sorted(os.listdir(path))
    if not should_compress(path) or len(items) <= 10:
        return items

    return items[:3] + ["..."] + items[-3:]

def get_structure(root_dir, prefix=""):
    lines = []
    items = list_with_compression(root_dir)

    for idx, item in enumerate(items):
        path = os.path.join(root_dir, item)
        if item in EXCLUDE_DIRS:
            continue

        if os.path.isfile(path) and is_excluded_file(item):
            continue

        connector = "└── " if idx == len(items) - 1 else "├── "

        if item == "...":
            lines.append(prefix + "│   " + "...")
            continue

        lines.append(prefix + connector + item)

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

    print(f"[INFO] Structure saved in {output_file}")