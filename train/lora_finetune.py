"""
Fusion 模型 LoRA/QLoRA 微调脚本

支持：
- 8B 模型：单卡 24GB 全参微调，8GB QLoRA
- 14B 模型：双卡 24GB 全参，单卡 16GB+ QLoRA
- 动态推理控制（Thinking Dial）
- DeepSpeed ZeRO-3 支持

使用方法：
    # 8B 模型，单卡 24GB
    python train/lora_finetune.py --model_size 8B --data_path data/example.json
    
    # 14B 模型，QLoRA，单卡 16GB
    python train/lora_finetune.py --model_size 14B --quantize --lora_rank 64

作者：朱子瞻
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionDataset(Dataset):
    """
    Fusion 训练数据集
    
    支持 Thinking Dial 标签（think_rank）
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        add_thinking_token: bool = True,
    ):
        """
        初始化数据集
        
        数据格式（JSON）：
            [
                {
                    "prompt": "解释量子纠缠",
                    "response": "量子纠缠是...",
                    "think_rank": 2  # 可选：推理深度 0-3
                },
                ...
            ]
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_thinking_token = add_thinking_token
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"✅ 加载数据集：{len(self.data)} 条样本")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt = item["prompt"]
        response = item["response"]
        think_rank = item.get("think_rank", 0)  # 默认 0
        
        # 注入 Thinking Dial 控制 token
        if self.add_thinking_token and think_rank > 0:
            thinking_token = f"<|think| depth={think_rank}|>"
            full_text = f"{thinking_token}\n{prompt}\n{response}"
        else:
            full_text = f"{prompt}\n{response}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
        }


def create_model(
    model_size: str,
    quantize: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    """
    创建模型
    
    参数：
        model_size: "8B" 或 "14B"
        quantize: 是否量化（用于 QLoRA）
        load_in_4bit: 4-bit 量化（NF4）
        load_in_8bit: 8-bit 量化
    """
    model_name = f"fusion-{model_size.lower()}-base"
    
    logger.info(f"📦 加载模型：{model_name}")
    
    # 量化配置
    if quantize:
        if load_in_4bit:
            logger.info("🔧 使用 4-bit 量化（QLoRA）")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            model = prepare_model_for_kbit_training(model)
        elif load_in_8bit:
            logger.info("🔧 使用 8-bit 量化")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(model)
        else:
            raise ValueError("quantize=True 时必须指定 load_in_4bit 或 load_in_8bit")
    else:
        logger.info("🔧 全精度加载")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    return model


def apply_lora(
    model,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    target_modules: List[str] = None,
):
    """
    应用 LoRA 适配器
    
    参数：
        lora_rank: LoRA 秩（默认 64）
        lora_alpha: LoRA alpha（默认 16）
        target_modules: 目标模块（默认 q_proj, v_proj）
    """
    if target_modules is None:
        # 默认目标模块（根据模型架构调整）
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    logger.info(f"🔧 应用 LoRA（rank={lora_rank}, alpha={lora_alpha}）")
    logger.info(f"🔧 目标模块：{target_modules}")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model


def train(args):
    """
    主训练函数
    """
    logger.info("🚀 开始训练 Fusion 模型")
    logger.info(f"📊 模型大小：{args.model_size}")
    logger.info(f"📊 量化：{args.quantize}")
    logger.info(f"📊 LoRA rank：{args.lora_rank}")
    
    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"fusion-{args.model_size.lower()}-base")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 创建模型
    model = create_model(
        model_size=args.model_size,
        quantize=args.quantize,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    
    # 3. 应用 LoRA
    if args.use_lora:
        model = apply_lora(
            model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
    
    # 4. 加载数据集
    train_dataset = FusionDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        report_to="tensorboard",
        # DeepSpeed 配置（如果启用）
        deepspeed=args.deepspeed if args.use_deepspeed else None,
    )
    
    # 6. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding="longest",
        ),
    )
    
    # 7. 开始训练
    logger.info("🏃 开始训练...")
    trainer.train()
    
    # 8. 保存模型
    logger.info(f"💾 保存模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("✅ 训练完成！")


def main():
    parser = argparse.ArgumentParser(description="Fusion 模型 LoRA/QLoRA 微调")
    
    # 模型参数
    parser.add_argument("--model_size", type=str, default="8B", choices=["8B", "14B"],
                        help="模型大小（8B 或 14B）")
    parser.add_argument("--quantize", action="store_true",
                        help="是否使用量化（QLoRA）")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="4-bit 量化（NF4）")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="8-bit 量化")
    
    # LoRA 参数
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="是否使用 LoRA")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA 秩（rank）")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    
    # 训练参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据路径（JSON 格式）")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度")
    
    # 混合精度
    parser.add_argument("--fp16", action="store_true",
                        help="使用 FP16 混合精度")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="使用 BF16 混合精度（推荐）")
    
    # 日志和保存
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志打印间隔（步数）")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存检查点间隔（步数）")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="最多保存的检查点数")
    
    # DeepSpeed
    parser.add_argument("--use_deepspeed", action="store_true",
                        help="是否使用 DeepSpeed")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed 配置文件路径")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.quantize and not (args.load_in_4bit or args.load_in_8bit):
        raise ValueError("使用 --quantize 时必须指定 --load_in_4bit 或 --load_in_8bit")
    
    # 开始训练
    train(args)


if __name__ == "__main__":
    main()
