"""
模型评估指标

提供各种评估指标来计算模型性能：
- Perplexity (困惑度)
- BLEU score
- ROUGE score
- Accuracy
- Loss
"""

import math
import sys
import warnings
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EvaluationMetrics:
    """评估结果容器"""
    perplexity: float = 0.0
    loss: float = 0.0
    accuracy: float = 0.0
    bleu: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    
    def __str__(self) -> str:
        lines = ["[Evaluation Metrics]"]
        lines.append(f"  Perplexity: {self.perplexity:.4f}")
        lines.append(f"  Loss: {self.loss:.4f}")
        lines.append(f"  Accuracy: {self.accuracy:.4f}")
        lines.append(f"  BLEU: {self.bleu:.4f}")
        lines.append(f"  ROUGE-1: {self.rouge1:.4f}")
        lines.append(f"  ROUGE-2: {self.rouge2:.4f}")
        lines.append(f"  ROUGE-L: {self.rougeL:.4f}")
        return "\n".join(lines)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer = None,
        device: str = "cpu",
    ):
        """
        初始化评估器
        
        参数：
            model: 要评估的模型
            tokenizer: tokenizer（可选）
            device: 设备（cpu/cuda）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def compute_perplexity(
        self,
        text: str,
    ) -> float:
        """
        计算文本的困惑度（Perplexity）
        
        困惑度是语言模型的基本评估指标，
        表示模型对文本的预测能力（越低越好）。
        
        参数：
            text: 输入文本
            
        返回：
            困惑度值
        """
        if self.tokenizer is None:
            # Fallback: 使用 UTF-8 字节编码
            input_ids = torch.tensor([list(text.encode('utf-8'))], dtype=torch.long).to(self.device)
        else:
            input_ids = torch.tensor([self.tokenizer.encode(text)]).to(self.device)
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids[:, :-1],
            labels=input_ids[:, 1:],
        )
        
        # 计算困惑度
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs["loss"]
        # loss is already mean-reduced over tokens by model, so PPL = exp(loss)
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    @torch.no_grad()
    def compute_loss(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> float:
        """
        计算一批文本的平均 loss
        
        参数：
            texts: 文本列表
            max_length: 最大长度
            
        返回：
            平均 loss
        """
        total_loss = 0.0
        count = 0
        
        for text in texts:
            if self.tokenizer is None:
                input_ids = torch.tensor([list(text.encode('utf-8'))[:max_length]], dtype=torch.long).to(self.device)
            else:
                encoded = self.tokenizer.encode(text, max_length=max_length, truncation=True)
                input_ids = torch.tensor([encoded]).to(self.device)
            
            if input_ids.shape[1] < 2:
                continue  # 跳过太短的文本
            
            outputs = self.model(
                input_ids=input_ids[:, :-1],
                labels=input_ids[:, 1:],
            )
            
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            total_loss += loss.item()
            count += 1
        
        return total_loss / max(count, 1)
    
    @torch.no_grad()
    def compute_accuracy(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> float:
        """
        计算下一个 token 预测准确率
        
        参数：
            texts: 文本列表
            max_length: 最大长度
            
        返回：
            准确率（0-1）
        """
        correct = 0
        total = 0
        
        for text in texts:
            if self.tokenizer is None:
                input_ids = torch.tensor([list(text.encode('utf-8'))[:max_length]], dtype=torch.long).to(self.device)
            else:
                encoded = self.tokenizer.encode(text, max_length=max_length, truncation=True)
                input_ids = torch.tensor([encoded]).to(self.device)
            
            if input_ids.shape[1] < 2:
                continue
            
            outputs = self.model(
                input_ids=input_ids[:, :-1],
            )
            
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            predictions = logits.argmax(dim=-1)
            targets = input_ids[:, 1:]
            
            # 只计算有效位置
            correct += (predictions == targets).sum().item()
            total += targets.numel()
        
        return correct / max(total, 1)
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """
        计算 BLEU score（简化版）

        .. deprecated::
            请使用 evaluation/bleu_rouge_meteor.py:compute_bleu() 代替。
            该实现缺少 BLEU-4 brevity penalty 和 n-gram clipping。
        
        参数：
            predictions: 预测文本列表
            references: 参考文本列表
            
        返回：
            BLEU score（0-1）
        """
        if len(predictions) != len(references):
            raise ValueError("predictions 和 references 长度必须相同")
        
        total_bleu = 0.0
        
        for pred, ref in zip(predictions, references):
            # 简化的 BLEU：计算 1-gram 和 2-gram 重合度
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                continue
            
            # 1-gram 重合
            pred_unigram_set = set(pred_tokens)
            ref_unigram_set = set(ref_tokens)
            unigram_overlap = len(pred_unigram_set & ref_unigram_set) / max(len(pred_unigram_set), 1)
            
            # 2-gram 重合
            pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
            ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
            if len(pred_bigrams) > 0 and len(ref_bigrams) > 0:
                bigram_overlap = len(pred_bigrams & ref_bigrams) / max(len(pred_bigrams), 1)
            else:
                bigram_overlap = 0.0
            
            # BLEU 简化公式：0.5 * unigram + 0.5 * bigram
            bleu = 0.5 * unigram_overlap + 0.5 * bigram_overlap
            total_bleu += bleu
        
        return total_bleu / max(len(predictions), 1)
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        计算 ROUGE score（简化版）

        .. deprecated::
            请使用 evaluation/bleu_rouge_meteor.py:compute_rouge() 代替。
            该实现缺少标准 ROUGE-L 的 LCS 精确计算。
        
        参数：
            predictions: 预测文本列表
            references: 参考文本列表
            
        返回：
            ROUGE-1, ROUGE-2, ROUGE-L 的字典
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
                continue
            
            # ROUGE-1: unigram 重合率
            pred_unigram_set = set(pred_tokens)
            ref_unigram_set = set(ref_tokens)
            overlap_1 = len(pred_unigram_set & ref_unigram_set)
            rouge1 = overlap_1 / max(len(ref_unigram_set), 1)
            rouge1_scores.append(rouge1)
            
            # ROUGE-2: bigram 重合率
            pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
            ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
            if len(ref_bigrams) > 0:
                overlap_2 = len(pred_bigrams & ref_bigrams)
                rouge2 = overlap_2 / max(len(ref_bigrams), 1)
            else:
                rouge2 = 0.0
            rouge2_scores.append(rouge2)
            
            # ROUGE-L: 最长公共子序列（简化版：用编辑距离近似）
            lcs_length = self._approximate_lcs(pred_tokens, ref_tokens)
            rougeL = lcs_length / max(len(ref_tokens), 1)
            rougeL_scores.append(rougeL)
        
        return {
            "rouge1": sum(rouge1_scores) / max(len(rouge1_scores), 1),
            "rouge2": sum(rouge2_scores) / max(len(rouge2_scores), 1),
            "rougeL": sum(rougeL_scores) / max(len(rougeL_scores), 1),
        }
    
    def _approximate_lcs(self, seq1: List[str], seq2: List[str]) -> int:
        """近似计算最长公共子序列长度"""
        # 简化版：使用动态规划
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @torch.no_grad()
    def evaluate(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> EvaluationMetrics:
        """
        完整评估：计算所有指标
        
        参数：
            texts: 文本列表
            max_length: 最大长度
            
        返回：
            EvaluationMetrics 对象
        """
        metrics = EvaluationMetrics()
        
        # 1. Perplexity（使用第一个文本）
        if len(texts) > 0:
            metrics.perplexity = self.compute_perplexity(texts[0])
        
        # 2. Loss
        metrics.loss = self.compute_loss(texts, max_length)
        
        # 3. Accuracy
        metrics.accuracy = self.compute_accuracy(texts, max_length)
        
        return metrics


def evaluate_model(
    model: torch.nn.Module,
    texts: List[str],
    tokenizer = None,
    device: str = "cpu",
) -> EvaluationMetrics:
    """
    便捷函数：评估模型
    
    参数：
        model: 要评估的模型
        texts: 评估文本
        tokenizer: tokenizer（可选）
        device: 设备
        
    返回：
        评估指标
    """
    evaluator = ModelEvaluator(model, tokenizer, device)
    return evaluator.evaluate(texts)


if __name__ == "__main__":
    print("[Evaluation] 模型评估指标模块")
    print()
    print("功能：")
    print("  - Perplexity（困惑度）")
    print("  - Loss")
    print("  - Accuracy")
    print("  - BLEU score")
    print("  - ROUGE score")
    print()
    print("用法：")
    print("  from evaluation.metrics import ModelEvaluator")
    print("  evaluator = ModelEvaluator(model, tokenizer)")
    print("  metrics = evaluator.evaluate(texts)")
    print("  print(metrics)")
