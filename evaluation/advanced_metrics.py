"""
高级评估指标 - Distinct-n、Repetition Rate、Token Entropy 等
"""
import sys
import torch
import math
from collections import Counter
sys.path.insert(0, '.')


def distinct_n(generated_texts, n=1):
    """
    计算 Distinct-n（多样性指标）
    
    Args:
        generated_texts: 生成的文本列表（token IDs）
        n: n-gram 的 n
    
    Returns:
        float: Distinct-n 分数（0-1，越高越好）
    """
    total_ngrams = 0
    unique_ngrams = 0
    
    for text in generated_texts:
        # 转换为 n-gram
        ngrams = [tuple(text[i:i+n]) for i in range(len(text) - n + 1)]
        
        if len(ngrams) == 0:
            continue
        
        # 计数
        total_ngrams += len(ngrams)
        unique_ngrams += len(set(ngrams))
    
    if total_ngrams == 0:
        return 0.0
    
    # Distinct-n = unique_ngrams / total_ngrams
    return unique_ngrams / total_ngrams


def repetition_rate(generated_texts, n=3):
    """
    计算重复率（Repetition Rate）
    
    Args:
        generated_texts: 生成的文本列表（token IDs）
        n: n-gram 的 n
    
    Returns:
        float: 重复率（0-1，越低越好）
    """
    total_ngrams = 0
    repeated_ngrams = 0
    
    for text in generated_texts:
        # 转换为 n-gram
        ngrams = [tuple(text[i:i+n]) for i in range(len(text) - n + 1)]
        
        if len(ngrams) == 0:
            continue
        
        # 计数
        counter = Counter(ngrams)
        total_ngrams += len(ngrams)
        repeated_ngrams += sum(count for count in counter.values() if count > 1)
    
    if total_ngrams == 0:
        return 0.0
    
    # Repetition Rate = repeated_ngrams / total_ngrams
    return repeated_ngrams / total_ngrams


def average_length(generated_texts):
    """
    计算平均序列长度
    
    Args:
        generated_texts: 生成的文本列表（token IDs）
    
    Returns:
        float: 平均序列长度
    """
    if len(generated_texts) == 0:
        return 0.0
    
    total_length = sum(len(text) for text in generated_texts)
    return total_length / len(generated_texts)


def token_entropy(generated_texts, vocab_size):
    """
    计算 Token 熵（Entropy）
    
    Args:
        generated_texts: 生成的文本列表（token IDs）
        vocab_size: 词汇表大小
    
    Returns:
        float: Token 熵（越高表示多样性越好）
    """
    # 合并所有 token
    all_tokens = [token for text in generated_texts for token in text]
    
    if len(all_tokens) == 0:
        return 0.0
    
    # 计算概率分布
    counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    # 计算熵
    entropy = 0.0
    for token, count in counter.items():
        prob = count / total_tokens
        entropy -= prob * math.log(prob, 2)
    
    # 归一化到 0-1（最大熵 = log2(vocab_size)）
    max_entropy = math.log(vocab_size, 2)
    if max_entropy == 0:
        return 0.0
    
    return entropy / max_entropy


def evaluate_advanced(generated_texts, vocab_size, n=1):
    """
    评估高级指标
    
    Args:
        generated_texts: 生成的文本列表（token IDs）
        vocab_size: 词汇表大小
        n: n-gram 的 n
    
    Returns:
        dict: 高级指标字典
    """
    metrics = {}
    
    # Distinct-n
    metrics["distinct_n"] = distinct_n(generated_texts, n)
    
    # Repetition Rate
    metrics["repetition_rate"] = repetition_rate(generated_texts, n)
    
    # Average Length
    metrics["avg_length"] = average_length(generated_texts)
    
    # Token Entropy
    metrics["token_entropy"] = token_entropy(generated_texts, vocab_size)
    
    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 高级评估指标测试")
    print("=" * 60)
    print()
    
    # 创建示例生成文本
    generated_texts = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
        [1, 2, 8, 9, 10],
    ]
    vocab_size = 100
    
    # 评估
    metrics = evaluate_advanced(generated_texts, vocab_size, n=1)
    
    # 打印结果
    print("[METRICS] 高级评估指标:")
    print(f"   Distinct-1: {metrics['distinct_n']:.4f}")
    print(f"   Repetition Rate: {metrics['repetition_rate']:.4f}")
    print(f"   Average Length: {metrics['avg_length']:.2f}")
    print(f"   Token Entropy: {metrics['token_entropy']:.4f}")
    print()
    
    print("[PASS] 高级评估指标测试通过")
    sys.exit(0)
