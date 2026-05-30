#!/usr/bin/env python3
"""Fix mini_data.json: distribute think_rank 0-3 based on prompt content."""

import json
import re

# Keywords suggesting different thinking depths
DEPTH_3_KEYWORDS = ['prove', 'theorem', 'proof', 'derive', 'mathematical', 'complex',
                    'prove', 'derive', 'calculate', 'analyze deeply',
                    '\u8bc1\u660e', '\u63a8\u5bfc', '\u5b9a\u7406', '\u590d\u6742', '\u6df1\u5165\u5206\u6790']
DEPTH_2_KEYWORDS = ['explain', 'why', 'how does', 'compare', 'difference',
                    'algorithm', 'design', 'optimize',
                    '\u89e3\u91ca', '\u4e3a\u4ec0\u4e48', '\u5982\u4f55', '\u6bd4\u8f83', '\u7b97\u6cd5', '\u8bbe\u8ba1', '\u4f18\u5316']
DEPTH_1_KEYWORDS = ['write', 'implement', 'code', 'function', 'create',
                    '\u5199', '\u5b9e\u73b0', '\u7f16\u5199', '\u4ee3\u7801', '\u521b\u5efa']


def assign_depth(item):
    text = (item.get('prompt', '') + ' ' + item.get('response', '')).lower()
    for kw in DEPTH_3_KEYWORDS:
        if kw.lower() in text:
            return 3
    for kw in DEPTH_2_KEYWORDS:
        if kw.lower() in text:
            return 2
    for kw in DEPTH_1_KEYWORDS:
        if kw.lower() in text:
            return 1
    return 0


def main():
    with open('data/mini_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Count current distribution
    old_dist = {}
    for item in data:
        r = item.get('think_rank', 0)
        old_dist[r] = old_dist.get(r, 0) + 1
    print(f"Before fix: {old_dist}")

    # Fix
    for item in data:
        item['think_rank'] = assign_depth(item)

    # Count new distribution
    new_dist = {}
    for item in data:
        r = item.get('think_rank', 0)
        new_dist[r] = new_dist.get(r, 0) + 1
    print(f"After fix: {new_dist}")

    with open('data/mini_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fixed {len(data)} items")


if __name__ == '__main__':
    main()
