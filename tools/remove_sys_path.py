#!/usr/bin/env python3
"""Remove sys.path.insert/append lines that exist only for imports."""

import re
from pathlib import Path

def remove_sys_path_hacks(filepath):
    """Remove sys.path manipulation lines used for imports."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original = content
        
        # Pattern to match common sys.path hacks
        patterns = [
            r'import sys\n.*?sys\.path\.insert\([^)]+\)\n',  # import sys + insert
            r'sys\.path\.insert\([^)]+\)\n',  # standalone insert
            r'sys\.path\.append\([^)]+\)\n',  # standalone append
        ]
        
        # Also remove the project_root/repo_root variable definitions if they're only for sys.path
        # followed by sys.path.insert
        # Remove sys.path lines
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.MULTILINE)
        
        # Clean up orphaned project_root/repo_root definitions
        # Only remove if they appear before imports and aren't used elsewhere
        lines = content.split('\n')
        new_lines = []
        skip_next_blank = False
        
        for i, line in enumerate(lines):
            # Skip lines defining project_root/repo_root if not used elsewhere
                # Check if this variable is used elsewhere (not in sys.path)
                var_name = line.split('=')[0].strip()
                rest_of_file = '\n'.join(lines[i+1:])
                # Simple heuristic: skip if variable doesn't appear again
                if var_name not in rest_of_file:
                    skip_next_blank = True
                    continue
            if skip_next_blank and line.strip() == '':
                skip_next_blank = False
                continue
            
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        if content != original:
            filepath.write_text(content, encoding='utf-8')
            return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    return False

def main():
    root = Path('.')
    count = 0
    
    # Find all Python files with sys.path
    for pyfile in root.rglob('*.py'):
        # Skip .pixi and __pycache__ directories
        if '.pixi' in pyfile.parts or '__pycache__' in pyfile.parts:
            continue
        
        try:
            content = pyfile.read_text(encoding='utf-8')
            if 'sys.path' in content:
                if remove_sys_path_hacks(pyfile):
                    count += 1
                    print(f"Updated: {pyfile}")
        except Exception as e:
            pass
    
    print(f"\nTotal files updated: {count}")

if __name__ == '__main__':
    main()
