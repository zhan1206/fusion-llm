#!/usr/bin/env python3
"""Add depth=3 samples to mini_data.json for Thinking Dial coverage."""

import json

DEPTH_3_SAMPLES = [
    {"prompt": "Prove the Pythagorean theorem", "response": "Consider a right triangle with legs a, b and hypotenuse c. Construct a square of side (a+b) and arrange four copies of the triangle inside. The inner quadrilateral is a square of side c, so (a+b)^2 = c^2 + 4*(ab/2), which simplifies to a^2 + b^2 = c^2.", "think_rank": 3},
    {"prompt": "Derive the quadratic formula", "response": "Starting from ax^2 + bx + c = 0, divide by a, complete the square: (x + b/2a)^2 = (b^2 - 4ac)/4a^2. Taking the square root gives x = (-b +- sqrt(b^2 - 4ac)) / 2a.", "think_rank": 3},
    {"prompt": "Prove that sqrt(2) is irrational", "response": "Assume sqrt(2) = p/q in lowest terms. Then 2q^2 = p^2, so p is even. Write p = 2k, then 2q^2 = 4k^2, so q^2 = 2k^2, meaning q is also even. This contradicts p/q being in lowest terms.", "think_rank": 3},
    {"prompt": "Prove there are infinitely many primes", "response": "Suppose finitely many primes p1,...,pn exist. Let N = p1*p2*...*pn + 1. N is not divisible by any pi (remainder 1). So N is either prime itself or has a prime factor not in the list. Contradiction.", "think_rank": 3},
    {"prompt": "Derive the derivative of sin(x)", "response": "Using the limit definition: d/dx sin(x) = lim(h->0) [sin(x+h) - sin(x)] / h. Apply the angle addition formula: sin(x+h) = sin(x)cos(h) + cos(x)sin(h). This gives lim [sin(x)(cos(h)-1)/h + cos(x)sin(h)/h] = sin(x)*0 + cos(x)*1 = cos(x).", "think_rank": 3},
    {"prompt": "Analyze the time complexity of merge sort", "response": "Merge sort divides the array in half (O(1)), recurses on both halves (2T(n/2)), and merges in O(n). The recurrence T(n) = 2T(n/2) + O(n) solves to O(n log n) by the master theorem (case 2). This holds for worst, average, and best cases.", "think_rank": 3},
    {"prompt": "Prove the sum of first n natural numbers is n(n+1)/2", "response": "By induction: Base case n=1: 1 = 1(2)/2 = 1. Inductive step: assume S(k) = k(k+1)/2. Then S(k+1) = S(k) + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2. QED.", "think_rank": 3},
    {"prompt": "Prove that e^x converges for all x", "response": "The Taylor series e^x = sum(x^n/n!) has ratio test: |a_(n+1)/a_n| = |x|/(n+1) -> 0 as n -> infinity. Since the limit is 0 < 1 for all x, the series converges absolutely for all real x by the ratio test.", "think_rank": 3},
]

def main():
    with open('data/mini_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    old_dist = {}
    for item in data:
        r = item.get('think_rank', 0)
        old_dist[r] = old_dist.get(r, 0) + 1
    print(f"Before: {old_dist}")

    data.extend(DEPTH_3_SAMPLES)

    new_dist = {}
    for item in data:
        r = item.get('think_rank', 0)
        new_dist[r] = new_dist.get(r, 0) + 1
    print(f"After: {new_dist}")

    with open('data/mini_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Total: {len(data)} items")


if __name__ == '__main__':
    main()
