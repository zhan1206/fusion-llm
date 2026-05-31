#!/usr/bin/env python3
"""Fix think token format across the project.

Ensures all files use the unified <|think_depth_N|> format,
not the old <|think| depth=N|> format.
"""
import re
import sys
from pathlib import Path

# Old format patterns (to replace)
OLD_INLINE = re.compile(r'<\|think\|\s*depth=(\d+)\|>')
OLD_FSTRING = re.compile(r'f"<\|think\|\s*depth=\{(\w+)\}\|>"')
OLD_TEMPLATE = '<|think| depth='

# New format
NEW_INLINE = r'<|think_depth_\1|>'
NEW_FSTRING = r'f"<|think_depth_{\1}|>"'


def fix_file(filepath: Path) -> bool:
    """Fix think tokens in a single file. Returns True if changes were made."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception:
        return False
    
    original = content
    
    # Fix inline occurrences: <|think| depth=N|> -> <|think_depth_N|>
    content = OLD_INLINE.sub(NEW_INLINE, content)
    
    # Fix f-string templates: f"<|think| depth={var}|>" -> f"<|think_depth_{var}|>"
    content = OLD_FSTRING.sub(NEW_FSTRING, content)
    
    if content != original:
        filepath.write_text(content, encoding='utf-8')
        count = sum(1 for a, b in zip(original.split('\n'), content.split('\n')) if a != b)
        print(f"  Fixed {filepath} ({count} lines)")
        return True
    return False


def main():
    project_root = Path(__file__).parent.parent
    
    # Scan all Python files
    changed = 0
    for py_file in project_root.rglob('*.py'):
        if fix_file(py_file):
            changed += 1
    
    # Also check JSON data files
    for json_file in project_root.rglob('*.json'):
        if 'node_modules' in str(json_file):
            continue
        if fix_file(json_file):
            changed += 1
    
    print(f"\nTotal files fixed: {changed}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
