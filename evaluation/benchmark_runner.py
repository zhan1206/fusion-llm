"""
基准测试运行器
支持多种评估任务和配置
"""
import sys
import json
import time
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

sys.path.insert(0, '.')

from evaluation.metrics import ModelEvaluator, EvaluationMetrics
from evaluation.bertscore_moverscore import bertscore_simple, moverscore_simple


class BenchmarkRunner:
    """
    基准测试运行器
    
    支持的测试类型：
    - perplexity: 困惑度评估
    - generation: 生成质量评估
    - accuracy: 任务准确率
    - speed: 推理速度基准
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化基准测试运行器
        
        Args:
            model_path: 模型路径
            device: 计算设备 (auto/cpu/cuda)
        """
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.model = None
        self.config = None
        self.tokenizer = None
        
    def _resolve_device(self, device: str) -> str:
        """解析设备字符串"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self):
        """加载模型"""
        print(f"[Loading] Model from {self.model_path} on {self.device}")
        
        # 尝试加载 FusionMini
        try:
            from models.fusion_mini import FusionMini, FusionMiniConfig
            self.model = FusionMini._load_from_safetensors(self.model_path)
            self.config = self.model.config
            print(f"[Loaded] FusionMini model")
        except Exception as e:
            # 回退到 FusionModel
            from models.fusion_model import FusionModel, FusionConfig
            self.model = FusionModel.from_pretrained(self.model_path)
            self.config = self.model.config
            print(f"[Loaded] FusionModel: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 创建简单 tokenizer
        self.tokenizer = self._create_tokenizer()
        
    def _create_tokenizer(self):
        """创建简单 tokenizer（用于测试）"""
        vocab_size = getattr(self.config, 'vocab_size', 10000)
        
        class SimpleTokenizer:
            def __init__(self, vs):
                self.vocab_size = vs
                
            def encode(self, text):
                # 简单字符级编码
                return [ord(c) % self.vocab_size for c in text[:512]]
                
            def decode(self, ids):
                return ''.join(chr(i % 128 + 32) for i in ids if 0 <= i < self.vocab_size)
        
        return SimpleTokenizer(vocab_size)
    
    def run_perplexity(self, texts: List[str]) -> Dict[str, float]:
        """
        计算困惑度
        
        Args:
            texts: 测试文本列表
            
        Returns:
            困惑度指标
        """
        print(f"\n[Benchmark] Perplexity on {len(texts)} texts")
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                ids = self.tokenizer.encode(text)
                if len(ids) < 2:
                    continue
                    
                input_ids = torch.tensor([ids], device=self.device)
                labels = input_ids.clone()
                
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
                
                if loss is not None:
                    total_loss += loss.item() * len(ids)
                    total_tokens += len(ids)
        
        if total_tokens == 0:
            return {"perplexity": float('inf')}
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens
        }
    
    def run_generation_quality(
        self, 
        prompts: List[str], 
        references: List[str],
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """
        生成质量评估
        
        Args:
            prompts: 提示列表
            references: 参考答案列表
            max_new_tokens: 最大生成 token 数
            
        Returns:
            生成质量指标
        """
        print(f"\n[Benchmark] Generation quality on {len(prompts)} prompts")
        
        generations = []
        
        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                ids = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([ids], device=self.device)
                
                # 简单贪婪生成
                generated = input_ids.clone()
                for _ in range(max_new_tokens):
                    outputs = self.model(generated)
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    # EOS 检查（假设 2 是 EOS）
                    if next_token.item() == 2:
                        break
                
                gen_text = self.tokenizer.decode(generated[0].tolist())
                generations.append(gen_text)
        
        # 计算 BERTScore 和 MoverScore
        bert_scores = []
        mover_scores = []
        
        for gen, ref in zip(generations, references):
            gen_ids = self.tokenizer.encode(gen)
            ref_ids = self.tokenizer.encode(ref)
            
            _, _, bert_f1 = bertscore_simple(gen_ids, ref_ids)
            mover_score = moverscore_simple(gen_ids, ref_ids)
            
            bert_scores.append(bert_f1)
            mover_scores.append(mover_score)
        
        return {
            "bertscore_f1": sum(bert_scores) / len(bert_scores) if bert_scores else 0.0,
            "moverscore": sum(mover_scores) / len(mover_scores) if mover_scores else 0.0,
            "generations": generations[:5],  # 只返回前5个样本
            "num_samples": len(generations)
        }
    
    def run_speed_benchmark(
        self, 
        batch_sizes: List[int] = [1, 2, 4],
        seq_lengths: List[int] = [32, 64, 128, 256],
        warmup: int = 3,
        runs: int = 10
    ) -> Dict[str, Any]:
        """
        推理速度基准测试
        
        Args:
            batch_sizes: 批大小列表
            seq_lengths: 序列长度列表
            warmup: 预热次数
            runs: 测试次数
            
        Returns:
            速度指标
        """
        print(f"\n[Benchmark] Speed benchmark")
        
        results = []
        vocab_size = getattr(self.config, 'vocab_size', 10000)
        
        self.model.eval()
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                # 预热
                for _ in range(warmup):
                    dummy = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                    with torch.no_grad():
                        _ = self.model(dummy)
                
                # 计时
                torch.cuda.synchronize() if self.device == "cuda" else None
                start = time.perf_counter()
                
                for _ in range(runs):
                    dummy = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                    with torch.no_grad():
                        _ = self.model(dummy)
                
                torch.cuda.synchronize() if self.device == "cuda" else None
                end = time.perf_counter()
                
                avg_time = (end - start) / runs
                throughput = batch_size * seq_len / avg_time
                
                results.append({
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "latency_ms": avg_time * 1000,
                    "throughput_tokens_per_sec": throughput
                })
                
                print(f"  batch={batch_size}, seq={seq_len}: {avg_time*1000:.2f}ms")
        
        return {
            "results": results,
            "device": self.device,
            "runs": runs
        }
    
    def run_full_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行完整基准测试
        
        Args:
            config: 测试配置
            
        Returns:
            完整测试结果
        """
        print("="*60)
        print("Fusion-LLM Benchmark Runner")
        print("="*60)
        
        self.load_model()
        
        results = {
            "model_path": self.model_path,
            "device": self.device,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
        }
        
        # 困惑度
        if config.get("perplexity", True):
            test_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning models require large amounts of data.",
                "Natural language processing enables computers to understand text."
            ]
            results["perplexity"] = self.run_perplexity(test_texts)
        
        # 生成质量
        if config.get("generation", True):
            prompts = ["The future of AI is", "Machine learning helps"]
            references = ["The future of AI is bright and transformative.", "Machine learning helps solve complex problems."]
            results["generation"] = self.run_generation_quality(prompts, references)
        
        # 速度基准
        if config.get("speed", False):
            results["speed"] = self.run_speed_benchmark(
                batch_sizes=config.get("batch_sizes", [1]),
                seq_lengths=config.get("seq_lengths", [32, 64, 128])
            )
        
        print("\n[Benchmark] Complete")
        return results


def main():
    parser = argparse.ArgumentParser(description="Fusion-LLM Benchmark Runner")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    parser.add_argument("--perplexity", action="store_true", help="Run perplexity benchmark")
    parser.add_argument("--generation", action="store_true", help="Run generation benchmark")
    parser.add_argument("--speed", action="store_true", help="Run speed benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    
    args = parser.parse_args()
    
    # 配置
    config = {
        "perplexity": args.perplexity or args.all,
        "generation": args.generation or args.all,
        "speed": args.speed or args.all
    }
    
    # 运行
    runner = BenchmarkRunner(args.model, args.device)
    results = runner.run_full_benchmark(config)
    
    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Saved] Results to {args.output}")


if __name__ == "__main__":
    main()
