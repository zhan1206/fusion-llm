"""
BLEU、ROUGE、METEOR 评估指标
"""
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '.')


def compute_bleu(references, hypotheses, max_n=4):
    """
    计算 BLEU 分数
    
    Args:
        references: 参考文本列表（每个元素是字符串或token列表）
        hypotheses: 生成文本列表（每个元素是字符串或token列表）
        max_n: 最大 n-gram 阶数（默认4，即 BLEU-4）
    
    Returns:
        dict: {bleu_1, bleu_2, bleu_3, bleu_4, brevity_penalty}
    """
    def tokenize(text):
        if isinstance(text, str):
            return text.lower().split()
        return [t.lower() for t in text]
    
    def get_ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    bleu_scores = {}
    
    for n in range(1, max_n + 1):
        total_clip = 0
        total_count = 0
        
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = tokenize(ref)
            hyp_tokens = tokenize(hyp)
            
            ref_ngrams = get_ngrams(ref_tokens, n)
            hyp_ngrams = get_ngrams(hyp_tokens, n)
            
            clip = 0
            for ngram, count in hyp_ngrams.items():
                clip += min(count, ref_ngrams.get(ngram, 0))
            
            total_clip += clip
            total_count += max(sum(hyp_ngrams.values()), 1)
        
        if total_count == 0:
            bleu_scores[f'bleu_{n}'] = 0.0
        else:
            bleu_scores[f'bleu_{n}'] = total_clip / total_count
    
    # Brevity Penalty
    ref_lengths = [len(tokenize(r)) for r in references]
    hyp_lengths = [len(tokenize(h)) for h in hypotheses]
    
    bp = 1.0
    if sum(hyp_lengths) < sum(ref_lengths):
        bp = math.exp(1 - sum(ref_lengths) / max(sum(hyp_lengths), 1))
    
    # Combined BLEU score
    if all(bleu_scores[f'bleu_{n}'] > 0 for n in range(1, max_n + 1)):
        log_avg = sum(math.log(bleu_scores[f'bleu_{n}']) for n in range(1, max_n + 1)) / max_n
        bleu_scores['bleu'] = bp * math.exp(log_avg)
    else:
        bleu_scores['bleu'] = 0.0
    
    bleu_scores['brevity_penalty'] = bp
    
    return bleu_scores


def compute_rouge(references, hypotheses):
    """
    计算 ROUGE 分数（ROUGE-1, ROUGE-2, ROUGE-L）
    
    Args:
        references: 参考文本列表
        hypotheses: 生成文本列表
    
    Returns:
        dict: {rouge_1, rouge_2, rouge_l}（各含 precision, recall, f1）
    """
    def tokenize(text):
        if isinstance(text, str):
            return text.lower().split()
        return [t.lower() for t in text]
    
    def get_ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    def f1_score(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def lcs_length(x, y):
        """最长公共子序列长度"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    rouge_scores = {
        'rouge_1': {'precision': 0, 'recall': 0, 'f1': 0},
        'rouge_2': {'precision': 0, 'recall': 0, 'f1': 0},
        'rouge_l': {'precision': 0, 'recall': 0, 'f1': 0},
    }
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = tokenize(ref)
        hyp_tokens = tokenize(hyp)
        
        # ROUGE-1
        ref_unigrams = Counter(ref_tokens)
        hyp_unigrams = Counter(hyp_tokens)
        overlap = sum((ref_unigrams & hyp_unigrams).values())
        
        r1_p = overlap / max(len(hyp_tokens), 1)
        r1_r = overlap / max(len(ref_tokens), 1)
        rouge_scores['rouge_1']['precision'] += r1_p
        rouge_scores['rouge_1']['recall'] += r1_r
        rouge_scores['rouge_1']['f1'] += f1_score(r1_p, r1_r)
        
        # ROUGE-2
        ref_bigrams = get_ngrams(ref_tokens, 2)
        hyp_bigrams = get_ngrams(hyp_tokens, 2)
        overlap2 = sum((ref_bigrams & hyp_bigrams).values())
        
        r2_p = overlap2 / max(sum(hyp_bigrams.values()), 1)
        r2_r = overlap2 / max(sum(ref_bigrams.values()), 1)
        rouge_scores['rouge_2']['precision'] += r2_p
        rouge_scores['rouge_2']['recall'] += r2_r
        rouge_scores['rouge_2']['f1'] += f1_score(r2_p, r2_r)
        
        # ROUGE-L
        lcs = lcs_length(ref_tokens, hyp_tokens)
        rl_p = lcs / max(len(hyp_tokens), 1)
        rl_r = lcs / max(len(ref_tokens), 1)
        rouge_scores['rouge_l']['precision'] += rl_p
        rouge_scores['rouge_l']['recall'] += rl_r
        rouge_scores['rouge_l']['f1'] += f1_score(rl_p, rl_r)
    
    # 平均
    n = max(len(references), 1)
    for key in rouge_scores:
        for metric in rouge_scores[key]:
            rouge_scores[key][metric] /= n
    
    return rouge_scores


def compute_meteor(references, hypotheses, alpha=0.9, beta=3, gamma=0.5):
    """
    计算 METEOR 分数
    
    Args:
        references: 参考文本列表
        hypotheses: 生成文本列表
        alpha: 精确率权重（默认0.9）
        beta: 分段惩罚参数（默认3）
        gamma: 分段惩罚系数（默认0.5）
    
    Returns:
        dict: {meteor, precision, recall, fragmentation_penalty}
    """
    def tokenize(text):
        if isinstance(text, str):
            return text.lower().split()
        return [t.lower() for t in text]
    
    def align(hyp_tokens, ref_tokens):
        """简单的对齐：贪心匹配"""
        aligned = set()
        ref_used = set()
        
        for i, h_token in enumerate(hyp_tokens):
            for j, r_token in enumerate(ref_tokens):
                if j not in ref_used and h_token == r_token:
                    aligned.add((i, j))
                    ref_used.add(j)
                    break
        
        return aligned
    
    def count_chunks(aligned, hyp_tokens, ref_tokens):
        """计算块数（连续对齐段数）"""
        if not aligned:
            return 0
        
        # 按 hyp 索引排序
        sorted_align = sorted(aligned, key=lambda x: x[0])
        chunks = 1
        
        for i in range(1, len(sorted_align)):
            prev_h, prev_r = sorted_align[i-1]
            curr_h, curr_r = sorted_align[i]
            if curr_h != prev_h + 1 or curr_r != prev_r + 1:
                chunks += 1
        
        return chunks
    
    total_meteor = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = tokenize(ref)
        hyp_tokens = tokenize(hyp)
        
        # 对齐
        aligned = align(hyp_tokens, ref_tokens)
        
        # 精确率和召回率
        m = len(aligned)
        precision = m / max(len(hyp_tokens), 1)
        recall = m / max(len(ref_tokens), 1)
        
        # F-mean
        if precision + recall == 0:
            f_mean = 0
        else:
            f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        
        # 分段惩罚
        chunks = count_chunks(aligned, hyp_tokens, ref_tokens)
        if m > 0:
            frag = chunks / m
        else:
            frag = 1.0
        
        penalty = gamma * (frag ** beta)
        
        # METEOR
        meteor = f_mean * (1 - penalty)
        total_meteor += meteor
    
    n = max(len(references), 1)
    avg_meteor = total_meteor / n
    
    return {
        'meteor': avg_meteor,
        'note': f'alpha={alpha}, beta={beta}, gamma={gamma}'
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM BLEU/ROUGE/METEOR 评估指标测试")
    print("=" * 60)
    print()
    
    # 测试数据
    references = [
        "The cat sat on the mat and looked at the window",
        "Machine learning is a subset of artificial intelligence",
        "The weather is nice today and I want to go outside",
    ]
    
    hypotheses = [
        "The cat sat on the mat and looked outside",
        "Machine learning is a branch of artificial intelligence",
        "The weather is nice and I want to go out",
    ]
    
    # BLEU
    print("[1] BLEU 分数...")
    bleu = compute_bleu(references, hypotheses)
    for k, v in bleu.items():
        print(f"   {k}: {v:.4f}")
    print()
    
    # ROUGE
    print("[2] ROUGE 分数...")
    rouge = compute_rouge(references, hypotheses)
    for k, v in rouge.items():
        print(f"   {k}: P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f}")
    print()
    
    # METEOR
    print("[3] METEOR 分数...")
    meteor = compute_meteor(references, hypotheses)
    for k, v in meteor.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    print()
    
    # 边缘情况测试
    print("[4] 边缘情况测试...")
    
    # 空输入
    bleu_empty = compute_bleu([""], [""])
    assert bleu_empty['bleu'] == 0.0, "Empty BLEU should be 0"
    print("   空输入测试通过")
    
    # 完全匹配（需要足够长的句子形成4-gram）
    long_sentence = "the cat sat on the mat and looked at the window"
    bleu_match = compute_bleu([long_sentence], [long_sentence])
    assert bleu_match['bleu'] > 0, "Perfect match BLEU should be > 0"
    print(f"   完全匹配测试通过 (BLEU={bleu_match['bleu']:.4f})")
    
    # 无匹配
    rouge_nomatch = compute_rouge(["aaa bbb"], ["ccc ddd"])
    assert rouge_nomatch['rouge_1']['f1'] == 0.0, "No match ROUGE should be 0"
    print("   无匹配测试通过")
    
    print()
    print("[PASS] BLEU/ROUGE/METEOR 评估指标测试全部通过")
    sys.exit(0)
