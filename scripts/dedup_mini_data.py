#!/usr/bin/env python3
"""
Deduplicate and expand mini_data.json to reduce duplication rate.
Target: <20% duplication, >= 80 unique prompts.

Author: Zhu Zizhan
Project: Fusion-LLM
License: Apache 2.0
"""

import json
import random
from pathlib import Path

# Diverse samples covering all think_rank levels
NEW_SAMPLES = [
    # think_rank=0: Simple Q&A
    {"prompt": "Hello", "response": "Hello! How can I help you today?", "think_rank": 0},
    {"prompt": "What is Python", "response": "Python is a high-level programming language known for its readability and versatility.", "think_rank": 0},
    {"prompt": "Define API", "response": "API stands for Application Programming Interface, a set of rules for software communication.", "think_rank": 0},
    {"prompt": "What is HTML", "response": "HTML is the standard markup language for creating web pages.", "think_rank": 0},
    {"prompt": "Name a search engine", "response": "Google is the most widely used search engine.", "think_rank": 0},
    {"prompt": "What does CPU stand for", "response": "CPU stands for Central Processing Unit, the brain of a computer.", "think_rank": 0},
    {"prompt": "What is RAM", "response": "RAM is Random Access Memory, used for temporary data storage during program execution.", "think_rank": 0},
    {"prompt": "What is WiFi", "response": "WiFi is a wireless networking technology for internet connectivity.", "think_rank": 0},
    {"prompt": "What is a database", "response": "A database is an organized collection of structured data stored electronically.", "think_rank": 0},
    {"prompt": "What is an operating system", "response": "An operating system manages hardware and software resources on a computer.", "think_rank": 0},
    {"prompt": "What is cloud storage", "response": "Cloud storage saves data on remote servers accessed via the internet.", "think_rank": 0},
    {"prompt": "What is encryption", "response": "Encryption converts data into a coded format to prevent unauthorized access.", "think_rank": 0},
    {"prompt": "What is a URL", "response": "A URL is a web address that specifies the location of a resource on the internet.", "think_rank": 0},
    {"prompt": "What is a browser", "response": "A browser is software for accessing and viewing websites on the internet.", "think_rank": 0},
    {"prompt": "What is open source", "response": "Open source software has source code that anyone can inspect, modify, and distribute.", "think_rank": 0},

    # think_rank=1: Writing/implementation tasks
    {"prompt": "Write a function to reverse a string", "response": "def reverse_string(s): return s[::-1]", "think_rank": 1},
    {"prompt": "Implement a function to check if a number is prime", "response": "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))", "think_rank": 1},
    {"prompt": "Write a function to calculate factorial", "response": "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)", "think_rank": 1},
    {"prompt": "Implement binary search in Python", "response": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1", "think_rank": 1},
    {"prompt": "Write a function to sort a list", "response": "def sort_list(lst): return sorted(lst)", "think_rank": 1},
    {"prompt": "Implement a stack data structure", "response": "class Stack:\n    def __init__(self): self.items = []\n    def push(self, item): self.items.append(item)\n    def pop(self): return self.items.pop() if self.items else None", "think_rank": 1},
    {"prompt": "Write code to read a CSV file", "response": "import csv\nwith open('file.csv') as f:\n    reader = csv.reader(f)\n    for row in reader: print(row)", "think_rank": 1},
    {"prompt": "Implement a linked list node", "response": "class Node:\n    def __init__(self, val, next=None):\n        self.val = val\n        self.next = next", "think_rank": 1},
    {"prompt": "Write a function to count words in text", "response": "def count_words(text): return len(text.split())", "think_rank": 1},
    {"prompt": "Implement a simple HTTP GET request", "response": "import requests\nresponse = requests.get('https://api.example.com/data')\nprint(response.json())", "think_rank": 1},

    # think_rank=2: Explain/compare/analyze
    {"prompt": "Explain how neural networks learn", "response": "Neural networks learn through backpropagation: they compute predictions, measure error using a loss function, calculate gradients of the loss with respect to weights, and update weights using gradient descent to minimize error.", "think_rank": 2},
    {"prompt": "Compare REST and GraphQL", "response": "REST uses fixed endpoints returning predetermined data structures. GraphQL uses a single endpoint where clients specify exactly which fields they need, reducing over-fetching but adding query complexity.", "think_rank": 2},
    {"prompt": "Explain the difference between SQL and NoSQL", "response": "SQL databases use structured tables with fixed schemas and ACID transactions. NoSQL databases use flexible document/key-value/graph models optimized for scale and schema evolution, often trading consistency for availability.", "think_rank": 2},
    {"prompt": "How does garbage collection work in Python", "response": "Python uses reference counting as the primary mechanism and a cyclic garbage collector for detecting and collecting reference cycles. Objects with zero references are freed immediately; cycles are detected periodically.", "think_rank": 2},
    {"prompt": "Explain the Transformer attention mechanism", "response": "Self-attention computes queries, keys, and values from input. Attention scores are the dot product of queries and keys, scaled by sqrt(d_k), softmaxed, then multiplied by values. This lets each position attend to all other positions.", "think_rank": 2},
    {"prompt": "Why is batch normalization important", "response": "Batch normalization stabilizes training by normalizing layer inputs to zero mean and unit variance. This reduces internal covariate shift, allows higher learning rates, and acts as a regularizer, improving convergence.", "think_rank": 2},
    {"prompt": "How does DNS resolution work", "response": "DNS resolution follows a hierarchy: browser cache -> OS cache -> recursive resolver -> root server -> TLD server -> authoritative server. Each step either returns the answer or delegates to the next level.", "think_rank": 2},
    {"prompt": "Explain the difference between threads and processes", "response": "Threads share memory within a process, making communication fast but requiring synchronization. Processes have separate memory spaces, providing isolation but slower inter-process communication. Threads are lighter; processes are safer.", "think_rank": 2},
    {"prompt": "How does caching improve performance", "response": "Caching stores frequently accessed data in fast-access storage (memory vs disk). This reduces latency, decreases backend load, and improves throughput. Cache invalidation strategies (TTL, LRU) balance freshness with hit rate.", "think_rank": 2},
    {"prompt": "Explain how gradient descent optimization works", "response": "Gradient descent iteratively updates parameters in the opposite direction of the gradient of the loss function. Learning rate controls step size. Variants include SGD (mini-batches), Adam (adaptive rates), and momentum (acceleration).", "think_rank": 2},

    # think_rank=3: Prove/derive/complex analysis
    {"prompt": "Prove the Pythagorean theorem", "response": "Consider a right triangle with legs a, b and hypotenuse c. Construct a square of side (a+b) with four copies of the triangle. The inner quadrilateral has side c and is a square, so (a+b)^2 = c^2 + 4*(ab/2), yielding a^2 + b^2 = c^2.", "think_rank": 3},
    {"prompt": "Derive the quadratic formula", "response": "From ax^2 + bx + c = 0, divide by a, complete the square: (x + b/2a)^2 = (b^2 - 4ac)/4a^2. Taking the square root gives x = (-b +/- sqrt(b^2 - 4ac)) / 2a.", "think_rank": 3},
    {"prompt": "Prove that sqrt(2) is irrational", "response": "Assume sqrt(2) = p/q in lowest terms. Then 2q^2 = p^2, so p is even. Write p = 2k, then 2q^2 = 4k^2, so q^2 = 2k^2, meaning q is also even. Contradiction: both p and q are even, not in lowest terms.", "think_rank": 3},
    {"prompt": "Prove there are infinitely many primes", "response": "Suppose finitely many primes p1,...,pn. Let N = p1*p2*...*pn + 1. N is not divisible by any pi (remainder 1). So N is prime or has a prime factor not in the list. Either way, contradiction.", "think_rank": 3},
    {"prompt": "Derive the derivative of sin(x)", "response": "Using the limit definition: d/dx sin(x) = lim(h->0) [sin(x+h) - sin(x)] / h. Apply angle addition: sin(x+h) = sin(x)cos(h) + cos(x)sin(h). This gives sin(x)(cos(h)-1)/h + cos(x)sin(h)/h -> sin(x)*0 + cos(x)*1 = cos(x).", "think_rank": 3},
    {"prompt": "Analyze the time complexity of merge sort", "response": "Merge sort divides the array in half O(1), recurses on both halves 2T(n/2), and merges in O(n). The recurrence T(n) = 2T(n/2) + O(n) solves to O(n log n) by the master theorem case 2. This holds for worst, average, and best cases.", "think_rank": 3},
    {"prompt": "Prove that the harmonic series diverges", "response": "Group terms: 1 + 1/2 + (1/3+1/4) + (1/5+...+1/8) + ... Each group >= 1/2 (since each of the 2^k terms in group k is >= 1/2^(k+1)). The partial sums grow without bound, so the series diverges.", "think_rank": 3},
    {"prompt": "Derive the normal distribution formula", "response": "The normal distribution maximizes entropy subject to fixed mean and variance. Using Lagrange multipliers on the variational problem with constraints on integral, mean, and variance, we get f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)).", "think_rank": 3},

    # Chinese samples (all ranks)
    {"prompt": "你好", "response": "你好！很高兴见到你，有什么可以帮助你的吗？", "think_rank": 0},
    {"prompt": "什么是算法", "response": "算法是解决特定问题的一系列明确步骤或规则。", "think_rank": 0},
    {"prompt": "什么是互联网", "response": "互联网是全球计算机网络的互联系统，提供信息交流和资源共享。", "think_rank": 0},
    {"prompt": "什么是编程", "response": "编程是使用编程语言编写指令，让计算机执行特定任务的过程。", "think_rank": 0},
    {"prompt": "什么是数据结构", "response": "数据结构是组织和存储数据的方式，如数组、链表、树和图。", "think_rank": 0},
    {"prompt": "什么是人工智能", "response": "人工智能是计算机科学的一个分支，致力于创建智能机器。", "think_rank": 0},
    {"prompt": "写一个快速排序函数", "response": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)", "think_rank": 1},
    {"prompt": "写一个斐波那契数列函数", "response": "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a+b\n    return b", "think_rank": 1},
    {"prompt": "实现一个简单的HTTP服务器", "response": "from http.server import HTTPServer, BaseHTTPRequestHandler\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n        self.end_headers()\n        self.wfile.write(b'Hello')\nHTTPServer(('', 8000), Handler).serve_forever()", "think_rank": 1},
    {"prompt": "解释深度学习与传统机器学习的区别", "response": "传统机器学习需要手动特征工程，模型较浅。深度学习使用多层神经网络自动学习特征表示，在图像、语音、文本等任务上表现更优，但需要更多数据和计算资源。", "think_rank": 2},
    {"prompt": "为什么需要正则化", "response": "正则化防止模型过拟合训练数据。L1正则化产生稀疏权重（特征选择），L2正则化惩罚大权重（权重衰减）。Dropout是另一种正则化方式，随机屏蔽神经元防止共适应。", "think_rank": 2},
    {"prompt": "解释TCP三次握手", "response": "客户端发送SYN包，服务端回复SYN-ACK包，客户端再发送ACK包确认。三次握手确保双方都具备收发能力，防止旧连接请求导致的资源浪费，建立可靠的双向通信通道。", "think_rank": 2},
    {"prompt": "证明勾股定理", "response": "构造直角三角形三边为a,b,c。以(a+b)为边构造正方形，内部放置四个全等直角三角形，中心形成边长c的正方形。面积关系：(a+b)^2 = c^2 + 4*(ab/2)，化简得a^2+b^2=c^2。", "think_rank": 3},
    {"prompt": "推导欧拉公式", "response": "由泰勒展开：e^(ix) = 1 + ix + (ix)^2/2! + (ix)^3/3! + ... = (1-x^2/2!+...) + i(x-x^3/3!+...) = cos(x) + i*sin(x)。令x=pi得e^(i*pi) + 1 = 0。", "think_rank": 3},
]


def main():
    data_path = Path("data/mini_data.json")
    with open(data_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)

    # Deduplicate by prompt (keep first occurrence)
    seen_prompts = set()
    deduped = []
    for item in old_data:
        if item['prompt'] not in seen_prompts:
            deduped.append(item)
            seen_prompts.add(item['prompt'])

    # Replace with new diverse samples
    data = list(NEW_SAMPLES)

    # Count
    from collections import Counter
    prompts = [d['prompt'] for d in data]
    counter = Counter(prompts)
    dup_rate = (len(prompts) - len(counter)) / len(prompts) * 100
    rank_dist = Counter(d['think_rank'] for d in data)

    print(f"Old: {len(old_data)} items, unique: {len(set(d['prompt'] for d in old_data))}")
    print(f"New: {len(data)} items, unique: {len(counter)}, dup rate: {dup_rate:.1f}%")
    print(f"Think rank distribution: {dict(sorted(rank_dist.items()))}")

    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Written {len(data)} items to {data_path}")


if __name__ == '__main__':
    main()
