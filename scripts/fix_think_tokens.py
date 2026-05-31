#!/usr/bin/env python3
"""Fix think token naming consistency across the project."""
import re
import glob

# Target format: <|think_depth_0|>, <|think_depth_1|>, etc.

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Replace THINK_START/THINK_END constants
    content = content.replace('THINK_START = "<|think_depth_"', 'THINK_START = "<|think_depth_"')
    content = content.replace('THINK_END = "|>"', 'THINK_END = "|>"')
    
    # Replace build_think_token return
    content = content.replace(
        'return f"{THINK_START}{depth}{THINK_END}"',
        'return f"{THINK_START}{depth}{THINK_END}"'
    )
    
    # Replace THINK_DEPTH_PATTERN regex
    content = content.replace(
        'THINK_DEPTH_PATTERN = re.compile(r"<\\|think\\| depth=(\\d+)\\|>")',
        'THINK_DEPTH_PATTERN = re.compile(r"<\\|think_depth_(\\d+)\\|>")'
    )
    
    # Replace any inline <|think| depth=N|> with <|think_depth_N|>
    content = re.sub(r'<\|think\|\s*depth=(\d+)\|>', r'<|think_depth_\1|>', content)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Fixed: {filepath}")
        return True
    else:
        print(f"  No change: {filepath}")
        return False

files = glob.glob("**/*.py", recursive=True) + glob.glob("**/*.json", recursive=True)
fixed = 0
for f in sorted(files):
    # Skip data files and output
    if any(skip in f for skip in ['node_modules', '.git', 'output/']):
        continue
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            text = fh.read()
        if '<|think|' in text or 'think| depth=' in text:
            if fix_file(f):
                fixed += 1
    except:
        pass

print(f"\nTotal files fixed: {fixed}")
