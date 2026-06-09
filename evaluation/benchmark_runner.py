"""
基准测试运行器
支持多种评估任务和配置，包括标准学术 benchmark
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
    - generation: 生成质量评估（BERTScore/MoverScore）
    - mmlu: MMLU 多任务语言理解
    - gsm8k: GSM8K 数学推理
    - speed: 推理速度基准
    """
    
    def __init__(self, model_path: str, device: str = "auto", tokenizer_path: Optional[str] = None):
        """
        初始化基准测试运行器
        
        Args:
            model_path: 模型路径
            device: 计算设备 (auto/cpu/cuda)
            tokenizer_path: tokenizer 路径（如有）
        """
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.model = None
        self.config = None
        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
        
    def _resolve_device(self, device: str) -> str:
        """解析设备字符串"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self):
        """加载模型和 tokenizer"""
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
        
        # 加载 tokenizer
        self._load_tokenizer()
        
    def _load_tokenizer(self):
        """加载 tokenizer（优先从路径加载，否则用简单 tokenizer）"""
        # 尝试加载真实 tokenizer
        if self.tokenizer_path and Path(self.tokenizer_path).exists():
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                print(f"[Tokenizer] Loaded from {self.tokenizer_path}")
                return
            except Exception as e:
                print(f"[Tokenizer] Failed to load from {self.tokenizer_path}: {e}")
        
        # 回退：尝试从模型路径加载
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"[Tokenizer] Loaded from model path {self.model_path}")
            return
        except Exception:
            pass
        
        # 最终回退：简单字符级 tokenizer
        print("[Tokenizer] Using simple character-level tokenizer")
        self.tokenizer = self._create_simple_tokenizer()
    
    def _create_simple_tokenizer(self):
        """创建简单 tokenizer（用于测试）"""
        vocab_size = getattr(self.config, 'vocab_size', 10000)
        
        class SimpleTokenizer:
            def __init__(self, vs):
                self.vocab_size = vs
                
            def encode(self, text):
                return [ord(c) % self.vocab_size for c in text[:512]]
                
            def decode(self, ids):
                return ''.join(chr(max(32, i % 128)) for i in ids if 0 <= i < self.vocab_size)
        
        return SimpleTokenizer(vocab_size)
    
    # ==================== Perplexity ====================
    
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
                if hasattr(ids, 'input_ids'):
                    ids = ids.input_ids
                if len(ids) < 2:
                    continue
                    
                input_ids = torch.tensor([ids], device=self.device)
                labels = input_ids.clone()
                
                outputs = self.model(input_ids=input_ids, labels=labels)
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
    
    # ==================== Generation Quality ====================
    
    def run_generation_quality(
        self, 
        prompts: List[str], 
        references: List[str],
        max_new_tokens: int = 50
    ) -> Dict[str, Any]:
        """
        生成质量评估（BERTScore + MoverScore）
        
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
                encoded = self.tokenizer.encode(prompt)
                if hasattr(encoded, 'input_ids'):
                    input_ids = torch.tensor(encoded.input_ids, device=self.device).unsqueeze(0)
                else:
                    input_ids = torch.tensor([encoded], device=self.device)
                
                # 简单贪婪生成
                generated = input_ids.clone()
                for _ in range(max_new_tokens):
                    outputs = self.model(input_ids=generated)
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    # EOS 检查
                    eos_token_id = getattr(self.tokenizer, 'eos_token_id', 2)
                    if next_token.item() == eos_token_id:
                        break
                
                gen_text = self.tokenizer.decode(generated[0].tolist())
                generations.append(gen_text)
        
        # 计算 BERTScore 和 MoverScore
        bert_scores = []
        mover_scores = []
        
        for gen, ref in zip(generations, references):
            gen_ids = self.tokenizer.encode(gen)
            ref_ids = self.tokenizer.encode(ref)
            if hasattr(gen_ids, 'input_ids'):
                gen_ids = gen_ids.input_ids
            if hasattr(ref_ids, 'input_ids'):
                ref_ids = ref_ids.input_ids
            
            _, _, bert_f1 = bertscore_simple(gen_ids, ref_ids)
            mover_score = moverscore_simple(gen_ids, ref_ids)
            
            bert_scores.append(bert_f1)
            mover_scores.append(mover_score)
        
        return {
            "bertscore_f1": sum(bert_scores) / len(bert_scores) if bert_scores else 0.0,
            "moverscore": sum(mover_scores) / len(mover_scores) if mover_scores else 0.0,
            "generations": generations[:5],
            "num_samples": len(generations)
        }
    
    # ==================== MMLU ====================
    
    def run_mmlu(self, num_samples: int = 100, few_shot: int = 5) -> Dict[str, Any]:
        """
        运行 MMLU 基准测试
        
        MMLU (Massive Multitask Language Understanding) 测试多任务语言理解能力。
        
        Args:
            num_samples: 每个子集采样数量（<=0 表示全部）
            few_shot: few-shot 示例数量
            
        Returns:
            MMLU 准确率等指标
        """
        print(f"\n[Benchmark] MMLU (few_shot={few_shot}, samples={num_samples})")
        
        try:
            from datasets import load_dataset
        except ImportError:
            print("[MMLU] datasets package not installed. pip install datasets")
            return {"error": "datasets not installed"}
        
        dataset = load_dataset("cais/mmlu", "all", split="test")
        
        if num_samples > 0:
            dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
        
        correct = 0
        total = 0
        results_by_subject = {}
        
        self.model.eval()
        with torch.no_grad():
            for sample in dataset:
                subject = sample["subject"]
                question = sample["question"]
                choices = sample["choices"]
                correct_answer = sample["answer"]
                
                # 构建 prompt
                prompt = self._build_mmlu_prompt(question, choices, few_shot)
                
                # 生成答案
                encoded = self.tokenizer.encode(prompt)
                if hasattr(encoded, 'input_ids'):
                    input_ids = torch.tensor(encoded.input_ids, device=self.device).unsqueeze(0)
                else:
                    input_ids = torch.tensor([encoded], device=self.device)
                
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
                
                # 解析答案（A/B/C/D）
                pred = self._parse_mmlu_answer(logits, prompt)
                is_correct = (pred == correct_answer)
                
                correct += int(is_correct)
                total += 1
                
                if subject not in results_by_subject:
                    results_by_subject[subject] = {"correct": 0, "total": 0}
                results_by_subject[subject]["correct"] += int(is_correct)
                results_by_subject[subject]["total"] += 1
        
        overall_acc = correct / total if total > 0 else 0.0
        
        # 按学科汇总
        subject_accs = {}
        for subj, stats in results_by_subject.items():
            subject_accs[subj] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return {
            "mmlu_overall_accuracy": overall_acc,
            "mmlu_total": total,
            "mmlu_subject_accuracy": subject_accs,
            "mmlu_num_subjects": len(subject_accs)
        }
    
    def _build_mmlu_prompt(self, question: str, choices: List[str], few_shot: int) -> str:
        """构建 MMLU few-shot prompt"""
        # 简化版：直接拼接问题和选项
        letters = ["A", "B", "C", "D"]
        choice_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:4]))
        prompt = f"Question: {question}\n{choice_text}\nAnswer:"
        return prompt
    
    def _parse_mmlu_answer(self, logits, prompt: str) -> int:
        """解析 MMLU 答案（A/B/C/D）"""
        # 简化版：检查 logits 中 A/B/C/D token 的概率
        # 实际应 tokenizer 编码 "A"/"B"/"C"/"D" 后取对应 logits
        answer_tokens = []
        for letter in ["A", "B", "C", "D"]:
            encoded = self.tokenizer.encode(letter)
            if hasattr(encoded, 'input_ids'):
                answer_tokens.append(encoded.input_ids[0])
            else:
                answer_tokens.append(encoded[0] if encoded else 0)
        
        # 取最后一个 token 的 logits（简化）
        last_logits = logits[0]
        answer_logits = [last_logits[t].item() for t in answer_tokens]
        return int(torch.tensor(answer_logits).argmax().item())
    
    # ==================== GSM8K ====================
    
    def run_gsm8k(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        运行 GSM8K 基准测试
        
        GSM8K (Grade School Math 8K) 测试小学数学推理能力。
        
        Args:
            num_samples: 采样数量（<=0 表示全部）
            
        Returns:
            GSM8K 准确率等指标
        """
        print(f"\n[Benchmark] GSM8K (samples={num_samples})")
        
        try:
            from datasets import load_dataset
        except ImportError:
            print("[GSM8K] datasets package not installed. pip install datasets")
            return {"error": "datasets not installed"}
        
        dataset = load_dataset("gsm8k", "main", split="test")
        
        if num_samples > 0:
            dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
        
        correct = 0
        total = 0
        gen_examples = []
        
        self.model.eval()
        with torch.no_grad():
            for sample in dataset:
                question = sample["question"]
                answer_text = sample["answer"]
                
                # 提取正确答案（GSM8K 答案格式：#### 数字）
                correct_answer = self._extract_gsm8k_answer(answer_text)
                
                # 生成
                prompt = f"Question: {question}\nAnswer:"
                encoded = self.tokenizer.encode(prompt)
                if hasattr(encoded, 'input_ids'):
                    input_ids = torch.tensor(encoded.input_ids, device=self.device).unsqueeze(0)
                else:
                    input_ids = torch.tensor([encoded], device=self.device)
                
                # 贪婪生成
                generated = input_ids.clone()
                for _ in range(128):
                    outputs = self.model(input_ids=generated)
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    eos_token_id = getattr(self.tokenizer, 'eos_token_id', 2)
                    if next_token.item() == eos_token_id:
                        break
                
                gen_text = self.tokenizer.decode(generated[0].tolist())
                pred_answer = self._extract_gsm8k_answer(gen_text)
                
                is_correct = (pred_answer == correct_answer)
                correct += int(is_correct)
                total += 1
                
                if len(gen_examples) < 3:
                    gen_examples.append({
                        "question": question[:100],
                        "pred": pred_answer,
                        "gold": correct_answer,
                        "correct": is_correct
                    })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "gsm8k_accuracy": accuracy,
            "gsm8k_total": total,
            "gsm8k_correct": correct,
            "gsm8k_examples": gen_examples
        }
    
    def _extract_gsm8k_answer(self, text: str) -> Optional[float]:
        """从 GSM8K 文本中提取数值答案"""
        import re
        # GSM8K 格式：#### 42
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # 回退：取最后一个数字
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None
    
    # ==================== HumanEval (Placeholder) ====================
    
    def run_humaneval(self, num_samples: int = 50) -> Dict[str, Any]:
        """
        运行 HumanEval 基准测试（占位实现）
        
        HumanEval 需要代码执行环境，此处提供框架，
        完整实现需要 integrate 代码执行沙箱。
        
        Args:
            num_samples: 采样数量
            
        Returns:
            占位结果
        """
        print(f"\n[Benchmark] HumanEval (samples={num_samples}) [PLACEHOLDER]")
        print("[HumanEval] Note: Full implementation requires code execution sandbox.")
        print("[HumanEval] Install: pip install human-eval")
        
        try:
            from datasets import load_dataset
        except ImportError:
            print("[HumanEval] datasets package not installed. pip install datasets")
            return {"error": "datasets not installed", "note": "placeholder"}
        
        dataset = load_dataset("openai_humaneval", split="test")
        
        if num_samples > 0:
            dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
        
        # 占位：记录 prompt，但不执行代码
        results = []
        for sample in dataset:
            results.append({
                "task_id": sample["task_id"],
                "prompt": sample["prompt"][:100],
                "note": "Code execution not implemented; install human-eval package for full evaluation"
            })
        
        return {
            "humaneval_samples": len(results),
            "humaneval_note": "Placeholder. Install `human-eval` and implement code execution for full evaluation.",
            "humaneval_samples_preview": results[:3]
        }
    
    # ==================== Speed Benchmark ====================
    
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
                        _ = self.model(input_ids=dummy)
                
                # 计时
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                for _ in range(runs):
                    dummy = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                    with torch.no_grad():
                        _ = self.model(input_ids=dummy)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
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
    
    # ==================== Full Benchmark ====================
    
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
        
        # MMLU
        if config.get("mmlu", False):
            results["mmlu"] = self.run_mmlu(
                num_samples=config.get("mmlu_samples", 100),
                few_shot=config.get("mmlu_few_shot", 5)
            )
        
        # GSM8K
        if config.get("gsm8k", False):
            results["gsm8k"] = self.run_gsm8k(
                num_samples=config.get("gsm8k_samples", 100)
            )
        
        # HumanEval
        if config.get("humaneval", False):
            results["humaneval"] = self.run_humaneval(
                num_samples=config.get("humaneval_samples", 50)
            )
        
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
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (optional)")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    
    # 测试项目开关
    parser.add_argument("--perplexity", action="store_true", help="Run perplexity benchmark")
    parser.add_argument("--generation", action="store_true", help="Run generation benchmark")
    parser.add_argument("--mmlu", action="store_true", help="Run MMLU benchmark")
    parser.add_argument("--gsm8k", action="store_true", help="Run GSM8K benchmark")
    parser.add_argument("--humaneval", action="store_true", help="Run HumanEval benchmark (placeholder)")
    parser.add_argument("--speed", action="store_true", help="Run speed benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    
    # MMLU/GSM8K 参数
    parser.add_argument("--mmlu-samples", type=int, default=100, help="MMLU samples per subject")
    parser.add_argument("--mmlu-few-shot", type=int, default=5, help="MMLU few-shot count")
    parser.add_argument("--gsm8k-samples", type=int, default=100, help="GSM8K samples")
    parser.add_argument("--humaneval-samples", type=int, default=50, help="HumanEval samples")
    
    args = parser.parse_args()
    
    # 配置
    config = {
        "perplexity": args.perplexity or args.all,
        "generation": args.generation or args.all,
        "mmlu": args.mmlu or args.all,
        "gsm8k": args.gsm8k or args.all,
        "humaneval": args.humaneval or args.all,
        "speed": args.speed or args.all,
        "mmlu_samples": args.mmlu_samples,
        "mmlu_few_shot": args.mmlu_few_shot,
        "gsm8k_samples": args.gsm8k_samples,
        "humaneval_samples": args.humaneval_samples,
    }
    
    # 运行
    runner = BenchmarkRunner(args.model, args.device, args.tokenizer)
    results = runner.run_full_benchmark(config)
    
    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n[Saved] Results to {args.output}")


if __name__ == "__main__":
    main()
