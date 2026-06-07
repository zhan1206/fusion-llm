"""
T-KD 教科书级知识蒸馏管道

使用开源教师模型（Qwen、DeepSeek等）对高信誉源（维基、教科书、学术论文）进行改写，
生成风格统一、论证清晰的教学文本。

注意：本脚本需要外部教师模型权重（如 Qwen2.5-72B-Instruct），这是蒸馏管道的设计要求。
若无法获取大型教师模型，可指定较小的本地模型路径替代，例如：
    --teacher_model "Qwen/Qwen2.5-7B-Instruct"  # 较小的替代方案
    --teacher_model "/path/to/local/model"       # 本地路径

使用方法：
    python data_pipeline/t_kd_distillation.py \
        --teacher_model "Qwen/Qwen2.5-72B-Instruct" \
        --data_sources "wikipedia,textbook,arxiv" \
        --output_path "data/t_kd_corpus.jsonl" \
        --num_samples 10000

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


class TKDDistiller:
    """
    教科书级知识蒸馏器
    
    使用教师模型生成高质量教学文本
    """
    
    def __init__(
        self,
        teacher_model: str,
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
    ):
        """
        初始化蒸馏器
        
        参数：
            teacher_model: 教师模型路径或 HuggingFace 模型 ID
            device: 设备（cuda/cpu）
            torch_dtype: 数据类型
        """
        print(f"[BOOK] 加载教师模型：{teacher_model}")
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            teacher_model,
            trust_remote_code=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            teacher_model,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        
        self.model.eval()
        
        print(f"[OK] 教师模型加载成功")
        print(f"   设备：{self.model.device}")
        print(f"   参数量：{sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        
    def generate_teaching_text(
        self,
        topic: str,
        source_type: str = "wikipedia",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """
        生成教学文本
        
        参数：
            topic: 主题（如 "量子纠缠"、"梯度下降"）
            source_type: 数据源类型（wikipedia/textbook/arxiv）
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            
        返回：
            生成的教学文本
        """
        # 构建提示词（根据数据源类型）
        if source_type == "wikipedia":
            prompt = f"""请以维基百科的风格，撰写一篇关于"{topic}"的详细条目。
要求：
1. 开头提供简明定义
2. 分段详细解释原理、历史、应用
3. 使用中立、学术的语言
4. 长度约 1500 字

条目内容："""
            
        elif source_type == "textbook":
            prompt = f"""请以大学教科书的风格，详细讲解"{topic}"。
要求：
1. 从基础概念开始，循序渐进
2. 包含关键公式（用 LaTeX 格式）
3. 提供实例或习题（附答案）
4. 使用清晰、教学性的语言
5. 长度约 2000 字

教学内容："""
            
        elif source_type == "arxiv":
            prompt = f"""请以学术论文的风格，撰写关于"{topic}"的研究综述。
要求：
1. 摘要（200字）
2. 引言（背景、意义）
3. 核心方法/理论（详细推导）
4. 实验/结果（假设性）
5. 结论与展望
6. 使用学术语言，包含参考文献（格式正确）
7. 长度约 2500 字

论文内容："""
            
        else:
            raise ValueError(f"不支持的 source_type：{source_type}")
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated_text.strip()
    
    def distill_batch(
        self,
        topics: List[str],
        source_type: str = "wikipedia",
        output_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        批量蒸馏
        
        参数：
            topics: 主题列表
            source_type: 数据源类型
            output_path: 输出文件路径（.jsonl）
            
        返回：
            蒸馏结果列表
        """
        print(f"\n[BOOK] 开始 T-KD 蒸馏...")
        print(f"   主题数：{len(topics)}")
        print(f"   数据源：{source_type}")
        
        results = []
        
        for i, topic in enumerate(topics):
            print(f"\n[{i+1}/{len(topics)}] 蒸馏主题：{topic}")
            
            try:
                # 生成教学文本
                text = self.generate_teaching_text(
                    topic=topic,
                    source_type=source_type,
                )
                
                # 保存结果
                result = {
                    "topic": topic,
                    "source_type": source_type,
                    "text": text,
                    "timestamp": time.time(),
                }
                
                results.append(result)
                
                # 实时保存（防止中断丢失）
                if output_path:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                print(f"[OK] 完成（生成 {len(text)} 字符）")
                
                # 避免 GPU 过热
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                    time.sleep(1)
                    
            except Exception as e:
                print(f"[FAIL] 失败：{e}")
                continue
        
        print(f"\n[DONE] 蒸馏完成！共生成 {len(results)} 个样本")
        
        return results


def load_topics(data_source: str) -> List[str]:
    """
    加载主题列表
    
    参数：
        data_source: 数据源（wikipedia/textbook/arxiv）
        
    返回：
        主题列表
    """
    # 预定义主题（实际应从知识库中提取）
    topics_map = {
        "wikipedia": [
            "量子纠缠",
            "Transformer 模型",
            "梯度下降",
            "卷积神经网络",
            "相对论",
            "机器学习",
            "区块链",
            "蛋白质折叠",
            "气候变化",
            "光合作用",
        ],
        "textbook": [
            "线性代数：特征分解",
            "概率论：贝叶斯定理",
            "微积分：链式法则",
            "统计学：假设检验",
            "信息论：熵与互信息",
            "优化理论：凸优化",
            "图论：最短路径算法",
            "数值分析：插值与拟合",
        ],
        "arxiv": [
            "大语言模型的上下文学习机制",
            "视觉-语言预训练方法综述",
            "图神经网络在分子性质预测中的应用",
            "自监督学习理论进展",
            "扩散模型在图像生成中的原理",
            "强化学习中的探索-利用权衡",
        ],
    }
    
    return topics_map.get(data_source, topics_map["wikipedia"])


def main():
    parser = argparse.ArgumentParser(
        description="T-KD 教科书级知识蒸馏"
    )
    
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct",
        help="教师模型（HuggingFace 模型 ID 或本地路径）",
    )
    
    parser.add_argument(
        "--data_sources",
        type=str,
        default="wikipedia,textbook,arxiv",
        help="数据源类型（逗号分隔）",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/t_kd_corpus.jsonl",
        help="输出文件路径（.jsonl）",
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="每个数据源生成的样本数",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备（cuda/cpu）",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("T-KD 教科书级知识蒸馏")
    print("=" * 60)
    
    # 1. 初始化蒸馏器
    distiller = TKDDistiller(
        teacher_model=args.teacher_model,
        device=args.device,
    )
    
    # 2. 解析数据源
    data_sources = [s.strip() for s in args.data_sources.split(",")]
    
    # 3. 对每个数据源进行蒸馏
    for source in data_sources:
        print(f"\n{'='*60}")
        print(f"数据源：{source}")
        print(f"{'='*60}")
        
        # 加载主题
        topics = load_topics(source)[:args.num_samples]
        
        # 蒸馏
        output_path = args.output_path.replace(".jsonl", f"_{source}.jsonl")
        results = distiller.distill_batch(
            topics=topics,
            source_type=source,
            output_path=output_path,
        )
        
        print(f"\n[OK] {source} 蒸馏完成，结果保存至：{output_path}")
    
    print(f"\n[DONE] 所有数据源蒸馏完成！")
    print(f"   输出目录：{Path(args.output_path).parent}")


if __name__ == "__main__":
    main()
