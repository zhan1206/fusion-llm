"""
Fusion 模型全参数微调脚本

支持：
- 本地 FusionModel（无需预训练权重）
- 8B 模型：单卡 24GB（开启 ZeRO-3 offload）
- 14B 模型：双卡 24GB 或单卡 48GB
- DeepSpeed ZeRO-3 支持
- 混合精度训练（BF16/FP16）

使用方法：
    # 本地模型全参微调
    python train/full_finetune.py --local_model --data_path data/example_data.json

    # 8B 模型 + DeepSpeed ZeRO-3
    deepspeed train/full_finetune.py --local_model --model_size 8B --deepspeed configs/ds_zero3.json --data_path data/example_data.json

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import argparse
import torch
import torch.nn as nn
import deepspeed
from transformers import (
    get_linear_schedule_with_warmup,
)
from models.tokenizer import get_tokenizer, get_effective_vocab_size
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FusionModel, FusionConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 数据格式说明
# ============================================================
"""
训练数据格式（JSON）：
[
    {
        "prompt": "解释量子纠缠",
        "response": "量子纠缠是...",
        "think_rank": 2
    },
    ...
]
"""


class FusionFullFinetuneDataset(Dataset):
    """
    全参数微调数据集
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"[FusionFullFinetuneDataset] 加载数据集：{len(self.data)} 条样本")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item["prompt"]
        response = item["response"]
        think_rank = item.get("think_rank", 0)
        
        if think_rank > 0:
            thinking_token = f"<|think_depth_{think_rank}|>"
            full_text = f"{thinking_token}\n{prompt}\n{response}"
        else:
            full_text = f"{prompt}\n{response}"
        
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


def create_local_model(
    model_size: str = "8B",
    torch_dtype: torch.dtype = torch.bfloat16,
    vocab_size_override: Optional[int] = None,
):
    """
    创建本地 FusionModel（无需预训练权重）
    """
    model_configs = {
        "0.5B": dict(vocab_size=32000, hidden_size=2048, num_hidden_layers=16,
                     num_attention_heads=16, num_key_value_heads=8, intermediate_size=5504),
        "1.5B": dict(vocab_size=32000, hidden_size=3072, num_hidden_layers=24,
                     num_attention_heads=24, num_key_value_heads=8, intermediate_size=8192),
        "8B": dict(vocab_size=100000, hidden_size=4096, num_hidden_layers=32,
                   num_attention_heads=32, num_key_value_heads=8, intermediate_size=11008),
        "14B": dict(vocab_size=100000, hidden_size=5120, num_hidden_layers=40,
                    num_attention_heads=40, num_key_value_heads=8, intermediate_size=13824),
    }
    
    if model_size not in model_configs:
        raise ValueError(f"不支持的模型大小：{model_size}")
    
    config_dict = model_configs[model_size]
    
    # S3 fix: override vocab_size to match actual tokenizer
    if vocab_size_override is not None:
        config_dict['vocab_size'] = vocab_size_override
    
    common_config = dict(
        block_size=512,
        latent_dim=64,
        window_size=2048,
        sbla_mode="hybrid",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        enable_thinking_dial=True,
        num_thinking_depths=4,
    )
    
    config = FusionConfig(**config_dict, **common_config)
    
    # S3-fix: sync vocab_size to actual tokenizer if provided
    if vocab_size_override is not None and vocab_size_override != config.vocab_size:
        logger.warning(f"[S3-fix] Overriding model vocab_size: {config.vocab_size} -> {vocab_size_override}")
        config.vocab_size = vocab_size_override
    
    logger.info(f"[create_local_model] 创建 Fusion-{model_size}（随机初始化）")
    logger.info(f"  hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
                f"heads={config.num_attention_heads}")
    
    model = FusionModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[create_local_model] 参数总量：{total_params / 1e9:.2f}B")
    
    return model, config


def create_tokenizer(tokenizer_type: str = "fusion", vocab_size: int = 32000):
    """
    Create tokenizer using the unified tokenizer module.
    """
    logger.info(f"[create_tokenizer] Creating tokenizer: type={tokenizer_type}, vocab_size={vocab_size}")
    tokenizer = get_tokenizer(tokenizer_type, vocab_size=vocab_size)
    return tokenizer


def train(args):
    """
    主训练函数
    """
    logger.info("=" * 60)
    logger.info("[train] 开始全参数微调")
    logger.info(f"  模型大小：{args.model_size}")
    logger.info(f"  使用 DeepSpeed：{args.deepspeed is not None}")
    logger.info(f"  数据路径：{args.data_path}")
    logger.info("=" * 60)
    
    # 1. 设备设置
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[train] 单卡训练，设备：{device}")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        logger.info(f"[train] 分布式训练，local_rank：{args.local_rank}")
    
    # 2. 加载 tokenizer
    vocab_size_map = {"0.5B": 32000, "1.5B": 32000, "8B": 100000, "14B": 100000}
    tokenizer = create_tokenizer(vocab_size=vocab_size_map.get(args.model_size, 32000))
    
    # Sync vocab_size to actual tokenizer size to prevent index-out-of-range (S3)
    actual_vocab_size = len(tokenizer)
    if actual_vocab_size != vocab_size_map.get(args.model_size, 32000):
        logger.warning(f"[S3-fix] Vocab size mismatch: config={vocab_size_map.get(args.model_size, 32000)}, tokenizer={actual_vocab_size}. Syncing to tokenizer.")
        vocab_size_map[args.model_size] = actual_vocab_size
    
    # 3. 创建模型（本地随机初始化）
    model, config = create_local_model(args.model_size, torch_dtype=args.torch_dtype, vocab_size_override=actual_vocab_size)
    
    # 4. 加载数据集
    train_dataset = FusionFullFinetuneDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # 5. 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    # 6. 学习率调度器
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # 7. DeepSpeed 初始化
    if args.deepspeed:
        logger.info(f"[train] 使用 DeepSpeed：{args.deepspeed}")
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=args.deepspeed,
        )
    else:
        model = model.to(device)
        model_engine = None
    
    # 8. 训练循环
    logger.info("[train] 开始训练循环...")
    
    global_step = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"[train] Epoch {epoch + 1}/{args.num_epochs}")
        
        model.train()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if args.deepspeed:
                outputs = model_engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                model_engine.backward(loss)
                model_engine.step()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            if global_step % args.logging_steps == 0:
                logger.info(f"Step {global_step} | Loss: {loss.item():.4f} | "
                            f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 保存检查点
        if args.deepspeed:
            if model_engine.local_rank == 0:
                save_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
                model_engine.save_checkpoint(save_path)
        else:
            if args.local_rank in [-1, 0]:
                save_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                config_path = os.path.join(save_path, "fusion_config.json")
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config.to_dict(), f, indent=2)
        
        logger.info(f"[train] Epoch {epoch + 1} 完成，保存到 {args.output_dir}")
    
    logger.info("[train] 全参数微调完成！")


def main():
    parser = argparse.ArgumentParser(description="Fusion 模型全参数微调")
    
    # 模型参数
    parser.add_argument("--model_size", type=str, default="1.5B",
                        choices=["0.5B", "1.5B", "8B", "14B"],
                        help="模型大小")
    parser.add_argument("--local_model", action="store_true", default=True,
                        help="使用本地 FusionModel（默认）")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="模型精度")
    
    # 训练参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据路径（JSON 格式）")
    parser.add_argument("--output_dir", type=str, default="./output/fusion-full",
                        help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小（根据显存调整）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="预热步数比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度")
    
    # 硬件参数
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="用于分布式训练（由 torchrun 自动设置）")
    
    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed 配置文件路径")
    
    # 日志
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志打印间隔")
    
    args = parser.parse_args()
    
    # 设置 torch dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    args.torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)
    
    train(args)


if __name__ == "__main__":
    main()