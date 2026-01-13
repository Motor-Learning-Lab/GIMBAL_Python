#!/usr/bin/env python3
"""Replace all 'import gimbal' and 'from gimbal' with gimbal_pymc."""

import re
from pathlib import Path


def replace_imports_in_file(filepath):
    """Replace gimbal imports with gimbal_pymc in a single file."""
    try:
        content = filepath.read_text(encoding="utf-8")
        original = content

        # Replace patterns
        content = re.sub(r"\bfrom gimbal\.", "from gimbal_pymc.", content)
        content = re.sub(r"\bfrom gimbal\s", "from gimbal_pymc ", content)
        content = re.sub(r"\bimport gimbal\.", "import gimbal_pymc.", content)
        content = re.sub(r"\bimport gimbal\s", "import gimbal_pymc ", content)
        content = re.sub(
            r"\bimport gimbal$", "import gimbal_pymc", content, flags=re.MULTILINE
        )

        if content != original:
            filepath.write_text(content, encoding="utf-8")
            return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    return False


def main():
    root = Path(".")
    count = 0

    # Find all Python files
    for pyfile in root.rglob("*.py"):
        # Skip .pixi and __pycache__ directories
        if ".pixi" in pyfile.parts or "__pycache__" in pyfile.parts:
            continue

        if replace_imports_in_file(pyfile):
            count += 1
            print(f"Updated: {pyfile}")

    print(f"\nTotal files updated: {count}")


if __name__ == "__main__":
    main()
