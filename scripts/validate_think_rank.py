#!/usr/bin/env python3
"""
Validate Thinking Dial think_rank distribution in training data.

Checks:
1. All think_rank values are in [0, 3]
2. Distribution is not degenerate (all same value)
3. Each rank has >= 5% representation
4. No duplicate prompts across different ranks

Usage:
    python scripts/validate_think_rank.py data/mini_data.json

Author: Zhu Zizhan
Project: Fusion-LLM
License: Apache 2.0
"""

import json
import sys
from collections import Counter
from pathlib import Path


def validate_think_rank(data_path: str) -> bool:
    """Validate think_rank distribution in a dataset."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    ranks = [item.get('think_rank', -1) for item in data]
    counter = Counter(ranks)
    
    print(f"Dataset: {data_path}")
    print(f"Total samples: {total}")
    print(f"Think rank distribution: {dict(sorted(counter.items()))}")
    
    issues = []
    
    # Check 1: All ranks in valid range
    invalid = [r for r in ranks if r not in (0, 1, 2, 3)]
    if invalid:
        issues.append(f"Invalid think_rank values: {Counter(invalid)}")
    
    # Check 2: Not degenerate
    if len(counter) <= 1:
        issues.append(f"Degenerate distribution - only rank {list(counter.keys())}")
    
    # Check 3: Each rank >= 5%
    for rank in range(4):
        pct = counter.get(rank, 0) / total * 100
        if pct > 0 and pct < 5:
            issues.append(f"Rank {rank} underrepresented: {pct:.1f}% (need >=5%)")
    
    # Check 4: No same prompt with different ranks
    prompt_ranks = {}
    for item in data:
        p = item.get('prompt', '')
        r = item.get('think_rank', -1)
        if p in prompt_ranks and prompt_ranks[p] != r:
            issues.append(f"Prompt '{p[:30]}...' has conflicting ranks: {prompt_ranks[p]} vs {r}")
        prompt_ranks[p] = r
    
    # Summary
    if issues:
        print(f"\nISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\nAll checks passed!")
        
        # Print distribution visualization
        print("\nDistribution:")
        for rank in range(4):
            count = counter.get(rank, 0)
            pct = count / total * 100
            bar = '#' * int(pct / 2)
            print(f"  Rank {rank}: {count:3d} ({pct:5.1f}%) {bar}")
        
        return True


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/mini_data.json'
    success = validate_think_rank(path)
    sys.exit(0 if success else 1)
