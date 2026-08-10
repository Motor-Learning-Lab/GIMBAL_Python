#!/usr/bin/env python3
"""Remove sys.path lines from all Python files."""
import re
from pathlib import Path


def clean_file(filepath):
    try:
        lines = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = []
        i = 0
        modified = False

        while i < len(lines):
            line = lines[i]

            # Skip sys.path.insert/append lines
            if "sys.path" in line and ("insert" in line or "append" in line):
                modified = True
                i += 1
                continue

            # Skip import sys if next non-empty line is sys.path
            if line.strip() == "import sys":
                # Look ahead for sys.path
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                    modified = True
                    i += 1
                    continue

            # Skip project_root/repo_root definitions if they look like path setup
            if re.match(r"\s*(project_root|repo_root)\s*=", line):
                # Check if next significant line is sys.path
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and "sys.path" in lines[j]:
                    modified = True
                    i += 1
                    continue

            new_lines.append(line)
            i += 1

        if modified:
            filepath.write_text("".join(new_lines), encoding="utf-8")
            return True
    except Exception as e:
        print(f"Error in {filepath}: {e}")
    return False


root = Path(".")
count = 0
for pyfile in root.rglob("*.py"):
    if ".pixi" in pyfile.parts or "__pycache__" in pyfile.parts:
        continue
    if clean_file(pyfile):
        count += 1
        print(f"Cleaned: {pyfile}")

print(f"\nTotal: {count} files")
