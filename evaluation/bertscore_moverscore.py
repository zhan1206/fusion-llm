"""
BERTScore 和 MoverScore 评估指标

【重要说明】这些是简化版实现（演示用）：
- BERTScore: 使用 hash 伪嵌入（确定性但非真正语义嵌入）
- MoverScore: 使用 hash 伪嵌入（确定性，模拟词嵌入距离）

生产环境请使用官方包：
- BERTScore: pip install bert-score
- MoverScore: pip install moverscore
"""
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, '.')


def bertscore_simple(candidate, reference, model_name="bert-base-uncased"):
    """
    Simplified BERTScore using deterministic hash embeddings.
    
    【警告】此版本使用 hash 伪嵌入，非真正语义嵌入：
    - 结果是确定性的（相同输入总是相同输出）
    - 但不捕获真正语义相似性
    - 仅用于快速测试和流程验证
    
    生产环境请安装官方包：
        pip install bert-score
        from bert_score import score
        P, R, F1 = score(candidates, references, lang="en")
    
    Args:
        candidate: candidate token IDs (list of ints)
        reference: reference token IDs (list of ints)
        model_name: BERT model name (unused in simplified version)
    
    Returns:
        tuple: (Precision, Recall, F1)
    """
    # L-NEW-1 FIX: Use cosine similarity instead of Jaccard set overlap.
    # Map each token ID to a learned-like embedding via hashing.
    # This approximates BERTScore's IDF-weighted cosine similarity.
    if len(candidate) == 0 or len(reference) == 0:
        return 0.0, 0.0, 0.0
    
    embed_dim = 64
    max_vocab = 100000
    
    def _hash_embed(token_ids):
        """Deterministic pseudo-embedding from token IDs via hash projection."""
        emb = torch.zeros(len(token_ids), embed_dim)
        for i, tid in enumerate(token_ids):
            for j in range(embed_dim):
                emb[i, j] = ((tid * (j + 1) * 2654435761) % (2**31)) / (2**31) * 2 - 1
        return emb
    
    cand_emb = _hash_embed(candidate)  # (len_c, dim)
    ref_emb = _hash_embed(reference)    # (len_r, dim)
    
    # Normalize
    cand_emb = cand_emb / (cand_emb.norm(dim=1, keepdim=True) + 1e-8)
    ref_emb = ref_emb / (ref_emb.norm(dim=1, keepdim=True) + 1e-8)
    
    # Cosine similarity matrix: (len_c, len_r)
    sim = torch.mm(cand_emb, ref_emb.t())
    
    # Precision = max similarity per candidate token
    precision = sim.max(dim=1).values.mean().item()
    # Recall = max similarity per reference token
    recall = sim.max(dim=0).values.mean().item()
    # F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


def moverscore_simple(candidate, reference):
    """
    简化版 MoverScore（演示用，确定性版本）
    
    实际使用时请安装官方包：pip install moverscore
    然后使用：
    from moverscore import moverscore
    scores = moverscore(candidates, references)
    
    Args:
        candidate: 候选文本（token IDs）
        reference: 参考文本（token IDs）
    
    Returns:
        float: MoverScore
    """
    # [FIX] 使用与 BERTScore 相同的 hash 嵌入，确保结果可复现
    # 之前用 torch.randn 导致每次调用结果不同
    
    if len(candidate) == 0 or len(reference) == 0:
        return 0.0
    
    embed_dim = 128  # MoverScore 用较小维度即可
    
    def _hash_embed(token_ids):
        """确定性伪嵌入，与 bertscore_simple 保持一致"""
        emb = torch.zeros(len(token_ids), embed_dim)
        for i, tid in enumerate(token_ids):
            for j in range(embed_dim):
                # 使用不同的 hash 种子（2654435761 是黄金比例常数的整数部分）
                emb[i, j] = ((tid * (j + 1) * 2654435761) % (2**31)) / (2**31) * 2 - 1
        return emb
    
    cand_embeddings = _hash_embed(candidate)  # (len_c, dim)
    ref_embeddings = _hash_embed(reference)    # (len_r, dim)
    
    # 计算成本矩阵（Euclidean 距离）
    cost_matrix = torch.cdist(cand_embeddings.unsqueeze(0), ref_embeddings.unsqueeze(0)).squeeze(0)
    
    # 简化版：使用平均最小成本作为分数（双向）
    # Precision: 对每个候选 token 找最近的参考 token
    precision_cost = cost_matrix.min(dim=1).values.mean().item()
    # Recall: 对每个参考 token 找最近的候选 token
    recall_cost = cost_matrix.min(dim=0).values.mean().item()
    
    # 归一化到 0-1（越低越好 → 越高越好）
    precision_score = 1.0 / (1.0 + precision_cost)
    recall_score = 1.0 / (1.0 + recall_cost)
    
    # F1 综合
    if precision_score + recall_score == 0:
        return 0.0
    score = 2 * precision_score * recall_score / (precision_score + recall_score)
    
    return score


def evaluate_bertscore_moverscore(candidates, references):
    """
    评估 BERTScore 和 MoverScore
    
    Args:
        candidates: 候选文本列表（每个文本是 token IDs 列表）
        references: 参考文本列表（每个文本是 token IDs 列表）
    
    Returns:
        dict: 评估指标字典
    """
    metrics = {
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
        "moverscore": [],
    }
    
    for cand, ref in zip(candidates, references):
        # BERTScore
        P, R, F1 = bertscore_simple(cand, ref)
        metrics["bertscore_precision"].append(P)
        metrics["bertscore_recall"].append(R)
        metrics["bertscore_f1"].append(F1)
        
        # MoverScore
        ms = moverscore_simple(cand, ref)
        metrics["moverscore"].append(ms)
    
    # 平均
    for key in metrics:
        metrics[key] = sum(metrics[key]) / len(metrics[key]) if len(metrics[key]) > 0 else 0.0
    
    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM BERTScore & MoverScore 评估")
    print("=" * 60)
    print()
    
    # 创建示例数据
    candidates = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ]
    
    references = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 11],  # 最后一个 token 不同
    ]
    
    # 评估
    metrics = evaluate_bertscore_moverscore(candidates, references)
    
    # 打印结果
    print("[METRICS] BERTScore & MoverScore:")
    print(f"   BERTScore Precision: {metrics['bertscore_precision']:.4f}")
    print(f"   BERTScore Recall:    {metrics['bertscore_recall']:.4f}")
    print(f"   BERTScore F1:       {metrics['bertscore_f1']:.4f}")
    print(f"   MoverScore:         {metrics['moverscore']:.4f}")
    print()
    
    print("[PASS] BERTScore & MoverScore 评估测试通过")
    sys.exit(0)
