"""Check all Python files for syntax errors."""

import ast
import pathlib
import sys


def check_syntax(file_path):
    """Check if a Python file has syntax errors."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            ast.parse(f.read())
        return None
    except SyntaxError as e:
        return f"{file_path}:{e.lineno}: {e.msg}"
    except Exception as e:
        return f"{file_path}: {type(e).__name__}: {e}"


errors = []
directories = ["gimbal_pymc", "tests", "examples", "debug", "tools"]

for directory in directories:
    dir_path = pathlib.Path(directory)
    if not dir_path.exists():
        continue
    for py_file in dir_path.rglob("*.py"):
        error = check_syntax(py_file)
        if error:
            errors.append(error)

if errors:
    print("Syntax errors found:")
    for error in errors:
        print(f"  {error}")
    sys.exit(1)
else:
    print("✓ No syntax errors found in any Python files")
