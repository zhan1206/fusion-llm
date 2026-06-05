#!/usr/bin/env python3
"""Manage mini_data.json: create, fix think_rank, add depth samples, dedup.

Usage:
    python manage_mini_data.py create   # Create initial dataset
    python manage_mini_data.py fix      # Re-assign think_rank by keywords
    python manage_mini_data.py enrich   # Add diverse samples for all depths
    python manage_mini_data.py dedup    # Remove duplicates
    python manage_mini_data.py all      # Run fix + enrich + dedup
"""

import json
import re
import sys
from pathlib import Path

DATA_PATH = Path("data/mini_data.json")

# --- Depth assignment keywords ---
DEPTH_3_KEYWORDS = ['prove', 'theorem', 'proof', 'derive', 'mathematical', 'complex',
                     'calculate', 'analyze deeply',
                     '\u8bc1\u660e', '\u63a8\u5bfc', '\u5b9a\u7406', '\u590d\u6742', '\u6df1\u5165\u5206\u6790']
DEPTH_2_KEYWORDS = ['explain', 'why', 'how does', 'compare', 'difference',
                     'algorithm', 'design', 'optimize',
                     '\u89e3\u91ca', '\u4e3a\u4ec0\u4e48', '\u5982\u4f55', '\u6bd4\u8f83', '\u7b97\u6cd5', '\u8bbe\u8ba1', '\u4f18\u5316']
DEPTH_1_KEYWORDS = ['write', 'implement', 'code', 'function', 'create',
                     '\u5199', '\u5b9e\u73b0', '\u7f16\u5199', '\u4ee3\u7801', '\u521b\u5efa']

# --- Extra diverse samples ---
EXTRA_SAMPLES = [
    # Depth 0 (simple)
    {"prompt": "What is 2+2?", "response": "4", "think_rank": 0},
    {"prompt": "Capital of France?", "response": "Paris", "think_rank": 0},
    {"prompt": "Hello", "response": "Hi! How can I help you?", "think_rank": 0},
    # Depth 1 (coding)
    {"prompt": "Write a function to reverse a string", "response": "def reverse_string(s): return s[::-1]", "think_rank": 1},
    {"prompt": "Write a function to calculate factorial", "response": "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)", "think_rank": 1},
    {"prompt": "Write a function to sort a list", "response": "def sort_list(lst): return sorted(lst)", "think_rank": 1},
    {"prompt": "Write code to read a CSV file", "response": "import csv\nwith open('file.csv') as f:\n    reader = csv.reader(f)\n    for row in reader: print(row)", "think_rank": 1},
    {"prompt": "Write a function to count words in text", "response": "def count_words(text): return len(text.split())", "think_rank": 1},
    # Depth 2 (reasoning)
    {"prompt": "Explain how neural networks learn", "response": "Neural networks learn by adjusting weights via backpropagation. The loss function measures prediction error; gradients indicate how each weight affects loss. An optimizer (e.g., SGD, Adam) updates weights in the direction that reduces loss, iteratively improving predictions.", "think_rank": 2},
    {"prompt": "Compare BFS and DFS graph traversal", "response": "BFS explores level by level using a queue (O(V+E)), guaranteeing shortest path in unweighted graphs. DFS goes deep before backtracking using a stack (O(V+E)), useful for topological sort and cycle detection. BFS uses more memory; DFS can be iterative or recursive.", "think_rank": 2},
    {"prompt": "Why does quicksort have O(n^2) worst case?", "response": "Quicksort's worst case occurs when the pivot is consistently the smallest or largest element, creating maximally unbalanced partitions. Each partition takes O(n) and we need n partitions, giving O(n^2). Median-of-three or random pivot selection mitigates this.", "think_rank": 2},
    # Depth 3 (math proof)
    {"prompt": "Prove that sqrt(2) is irrational", "response": "Assume sqrt(2) = p/q in lowest terms. Then 2q^2 = p^2, so p is even. Write p = 2k, then 2q^2 = 4k^2, so q^2 = 2k^2, meaning q is also even. This contradicts p/q being in lowest terms.", "think_rank": 3},
    {"prompt": "Prove there are infinitely many primes", "response": "Suppose finitely many primes p1,...,pn exist. Let N = p1*p2*...*pn + 1. N is not divisible by any pi (remainder 1). So N is either prime itself or has a prime factor not in the list. Contradiction.", "think_rank": 3},
    {"prompt": "Derive the quadratic formula", "response": "Starting from ax^2 + bx + c = 0, divide by a, complete the square: (x + b/2a)^2 = (b^2 - 4ac)/4a^2. Taking the square root gives x = (-b +- sqrt(b^2 - 4ac)) / 2a.", "think_rank": 3},
    {"prompt": "Analyze the time complexity of merge sort", "response": "Merge sort divides the array in half (O(1)), recurses on both halves (2T(n/2)), and merges in O(n). The recurrence T(n) = 2T(n/2) + O(n) solves to O(n log n) by the master theorem (case 2).", "think_rank": 3},
    {"prompt": "Prove the Pythagorean theorem", "response": "Consider a right triangle with legs a, b and hypotenuse c. Construct a square of side (a+b) and arrange four copies of the triangle inside. The inner quadrilateral is a square of side c, so (a+b)^2 = c^2 + 4*(ab/2), which simplifies to a^2 + b^2 = c^2.", "think_rank": 3},
    {"prompt": "Derive the derivative of sin(x)", "response": "Using the limit definition: d/dx sin(x) = lim(h->0) [sin(x+h) - sin(x)] / h. Apply the angle addition formula: sin(x+h) = sin(x)cos(h) + cos(x)sin(h). This gives lim [sin(x)(cos(h)-1)/h + cos(x)sin(h)/h] = sin(x)*0 + cos(x)*1 = cos(x).", "think_rank": 3},
    # Chinese samples
    {"prompt": "Explain the difference between TCP and UDP", "response": "TCP is connection-oriented, reliable, and ordered. UDP is connectionless, unreliable, and unordered. TCP handles retransmission and flow control, UDP does not. TCP is used for web, email; UDP for streaming, DNS.", "think_rank": 2},
    {"prompt": "Implement a simple HTTP server in Python", "response": "from http.server import HTTPServer, BaseHTTPRequestHandler\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n        self.end_headers()\n        self.wfile.write(b'Hello')\nHTTPServer(('', 8000), Handler).serve_forever()", "think_rank": 1},
]


def load_data():
    if DATA_PATH.exists():
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_data(data):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} items to {DATA_PATH}")


def show_dist(data):
    dist = {}
    for item in data:
        r = item.get('think_rank', 0)
        dist[r] = dist.get(r, 0) + 1
    print(f"Distribution: {dict(sorted(dist.items()))}")


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


def cmd_fix(data):
    for item in data:
        item['think_rank'] = assign_depth(item)
    print("Fixed think_rank distribution:")
    show_dist(data)
    return data


def cmd_enrich(data):
    existing_prompts = {item['prompt'] for item in data}
    added = 0
    for sample in EXTRA_SAMPLES:
        if sample['prompt'] not in existing_prompts:
            data.append(sample)
            existing_prompts.add(sample['prompt'])
            added += 1
    print(f"Added {added} new diverse samples")
    show_dist(data)
    return data


def cmd_dedup(data):
    seen = set()
    deduped = []
    for item in data:
        key = item.get('prompt', '')
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    removed = len(data) - len(deduped)
    print(f"Removed {removed} duplicates")
    return deduped


def cmd_create(data):
    """Create initial mini dataset."""
    print("Creating initial dataset...")
    return EXTRA_SAMPLES.copy()


def cmd_all(data):
    data = cmd_fix(data)
    data = cmd_enrich(data)
    data = cmd_dedup(data)
    return data


COMMANDS = {
    'create': cmd_create,
    'fix': cmd_fix,
    'enrich': cmd_enrich,
    'dedup': cmd_dedup,
    'all': cmd_all,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: {sys.argv[0]} [{'|'.join(COMMANDS)}]")
        sys.exit(1)

    cmd = sys.argv[1]
    data = load_data()
    print(f"Loaded {len(data)} items")
    show_dist(data)

    result = COMMANDS[cmd](data)
    save_data(result)


if __name__ == '__main__':
    main()
