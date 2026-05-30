"""
Fusion 模型全参数微调脚本

支持：
- 8B 模型：单卡 24GB（开启 ZeRO-3 offload）
- 14B 模型：双卡 24GB 或单卡 48GB
- DeepSpeed ZeRO-3 支持
- 混合精度训练（BF16/FP16）

使用方法：
    # 8B 模型，单卡 24GB
    deepspeed train/full_finetune.py --model_size 8B --deepspeed configs/ds_zero3.json
    
    # 14B 模型，双卡 24GB（DDP）
    torchrun --nproc_per_node=2 train/full_finetune.py --model_size 14B

作者：朱子瞻
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import argparse
import torch
import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import json
import os
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionFullFinetuneDataset(Dataset):
    """
    全参数微调数据集
    
    数据格式与 LoRA 相同，但支持更大批量
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
        
        logger.info(f"✅ 加载数据集：{len(self.data)} 条样本")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item["prompt"]
        response = item["response"]
        
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


def create_model(model_size: str, torch_dtype=torch.bfloat16):
    """
    创建模型（全参数）
    """
    model_name = f"fusion-{model_size.lower()}-base"
    
    logger.info(f"📦 加载模型（全参数）：{model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_cache=False,  # 训练时禁用 KV 缓存
    )
    
    return model


def train(args):
    """
    主训练函数
    """
    logger.info("🚀 开始全参数微调")
    logger.info(f"📊 模型大小：{args.model_size}")
    logger.info(f"📊 使用 DeepSpeed：{args.deepspeed is not None}")
    
    # 1. 初始化分布式训练（如果使用 torchrun）
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"💻 单卡训练，设备：{device}")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        logger.info(f"💻 分布式训练，local_rank：{args.local_rank}")
    
    # 2. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        f"fusion-{args.model_size.lower()}-base"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 创建模型
    model = create_model(args.model_size)
    
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
    
    # 7. DeepSpeed 初始化（如果启用）
    if args.deepspeed:
        logger.info(f"🔧 使用 DeepSpeed：{args.deepspeed}")
        
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=args.deepspeed,
        )
    else:
        model = model.to(device)
        model_engine = None
    
    # 8. 训练循环
    logger.info("🏃 开始训练...")
    
    global_step = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"📅 Epoch {epoch + 1}/{args.num_epochs}")
        
        model.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            if args.deepspeed:
                outputs = model_engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                
                # DeepSpeed 反向传播
                model_engine.backward(loss)
                model_engine.step()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                
                # 梯度累积
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            # 日志
            if global_step % args.logging_steps == 0:
                logger.info(f"Step {global_step} | Loss: {loss.item():.4f}")
            
            progress_bar.set_postfix({"loss": loss.item()})
        
        # 每个 epoch 保存一次
        if args.deepspeed:
            if model_engine.local_rank == 0:
                save_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
                model_engine.save_checkpoint(save_path)
        else:
            if args.local_rank in [-1, 0]:
                save_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
        
        logger.info(f"✅ Epoch {epoch + 1} 完成，模型保存到 {args.output_dir}")
    
    logger.info("✅ 训练完成！")


def main():
    parser = argparse.ArgumentParser(description="Fusion 模型全参数微调")
    
    # 模型参数
    parser.add_argument("--model_size", type=str, default="8B", choices=["8B", "14B"],
                        help="模型大小")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="模型精度")
    
    # 训练参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="./output",
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
    if args.torch_dtype == "float32":
        dtype = torch.float32
    elif args.torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    
    args.torch_dtype = dtype
    
    train(args)


if __name__ == "__main__":
    main()
