"""Standardize all imports to use 'import gimbal_pymc as gp'."""

import re
from pathlib import Path


def standardize_imports_in_file(filepath):
    """Standardize gimbal_pymc imports in a single file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Pattern 1: import gimbal_pymc as gimbal -> import gimbal_pymc as gp
    content = re.sub(
        r"^import gimbal_pymc as gimbal\b",
        "import gimbal_pymc as gp",
        content,
        flags=re.MULTILINE,
    )

    # Pattern 2: import gimbal_pymc -> import gimbal_pymc as gp (unless followed by from)
    # First check if the line is not malformed
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        # Match standalone "import gimbal_pymc" (not followed by other text on same line except comments/whitespace)
        if re.match(r"^import gimbal_pymc\s*(?:#.*)?$", line.strip()):
            new_lines.append(
                re.sub(
                    r"^(\s*)import gimbal_pymc\s*", r"\1import gimbal_pymc as gp ", line
                )
            )
        else:
            new_lines.append(line)
    content = "\n".join(new_lines)

    # Pattern 3: Fix malformed imports like "import gimbal_pymc from gimbal_pymc import"
    # Split into two lines
    content = re.sub(
        r"^import gimbal_pymc from gimbal_pymc import",
        "import gimbal_pymc as gp\nfrom gimbal_pymc import",
        content,
        flags=re.MULTILINE,
    )

    # Now replace usage patterns: gimbal. -> gp. (but not in comments or strings)
    # This is tricky - we'll do a simpler approach for now
    # Only replace "gimbal." when it's clearly a module reference

    # Pattern 4: Replace gimbal.function_name -> gp.function_name
    content = re.sub(r"\bgimbal\.([a-zA-Z_][a-zA-Z0-9_]*)", r"gp.\1", content)

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def main():
    """Process all Python files in the repository."""
    repo_root = Path(__file__).parent

    # Find all Python files
    python_files = list(repo_root.rglob("*.py"))

    # Exclude certain directories
    exclude_dirs = {".pixi", "__pycache__", ".git", "node_modules", ".venv"}
    python_files = [
        f for f in python_files if not any(exc in f.parts for exc in exclude_dirs)
    ]

    # Exclude this script itself and cleanup scripts
    exclude_files = {
        "standardize_imports.py",
        "rename_imports.py",
        "clean_sys_path.py",
        "remove_sys_path.py",
    }
    python_files = [f for f in python_files if f.name not in exclude_files]

    print(f"Processing {len(python_files)} Python files...")

    modified_count = 0
    modified_files = []

    for filepath in python_files:
        try:
            if standardize_imports_in_file(filepath):
                modified_count += 1
                modified_files.append(str(filepath.relative_to(repo_root)))
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print(f"\nModified {modified_count} files:")
    for f in modified_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
