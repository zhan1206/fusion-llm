"""
Fusion 模型 LoRA/QLoRA 微调脚本

支持：
- 本地 FusionModel（无需预训练权重）
- 8B 模型：单卡 24GB 全参微调，8GB QLoRA
- 14B 模型：双卡 24GB 全参，单卡 16GB+ QLoRA
- 动态推理控制（Thinking Dial）
- DeepSpeed ZeRO-3 支持

使用方法：
    # 本地模型训练（无需下载预训练权重）
    python train/lora_finetune.py --local_model --data_path data/example_data.json

    # 8B 模型 QLoRA
    python train/lora_finetune.py --local_model --model_size 8B --quantize --load_in_4bit --data_path data/example_data.json

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import sys
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# H8-H9: Wrap optional imports in try/except
try:
    import deepspeed
except ImportError:
    deepspeed = None
    logger.warning("DeepSpeed not installed. DeepSpeed features will be unavailable.")

try:
    import bitsandbytes
except ImportError:
    bitsandbytes = None

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# H4-H6: Use try/except for project imports
try:
    from models import FusionModel, FusionConfig
except ImportError:
    from models.fusion_model import FusionModel, FusionConfig



# ============================================================
# 数据格式说明
# ============================================================
"""
训练数据格式（JSON）：
[
    {
        "prompt": "解释量子纠缠",
        "response": "量子纠缠是...",
        "think_rank": 2  // 可选：推理深度 0-3，默认 0
    },
    ...
]

think_rank 说明：
- 0: 直接回答（闲聊、翻译、简单问答）
- 1: 简短思考后回答
- 2: 详细推理过程
- 3: 深度思考链
"""


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
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_thinking_token = add_thinking_token
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"[FusionDataset] 加载数据集：{len(self.data)} 条样本")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        item = self.data[idx]
        
        prompt = item["prompt"]
        response = item["response"]
        think_rank = item.get("think_rank", 0)
        
        # 注入 Thinking Dial 控制 token
        if self.add_thinking_token and think_rank > 0:
            thinking_token = f"<|think_depth_{think_rank}|>"
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


from train.model_utils import create_local_model as _create_local_model_from_utils


def create_local_model(
    model_size: str = "8B",
    quantize: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    vocab_size_override: int | None = None,
):
    """S4 FIX: Delegate to shared model_utils.create_local_model, then apply quantization."""
    model = _create_local_model_from_utils(
        model_size=model_size,
        torch_dtype=torch.bfloat16,
        vocab_size_override=vocab_size_override,
    )
    config = model.config

    # S-NEW-9 FIX: QLoRA requires proper bitsandbytes integration
    if quantize:
        if load_in_4bit:
            logger.info("[create_local_model] Using 4-bit quantization (QLoRA)")
            try:
                import bitsandbytes as bnb
                name_to_module = dict(model.named_modules())
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and not any(x in name for x in ['lora', 'head', 'embed']):
                        quantized = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            quant_type='nf4',
                            compute_dtype=torch.float16
                        )
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        if parent_name:
                            parent = name_to_module[parent_name]
                            setattr(parent, child_name, quantized)
                        else:
                            setattr(model, child_name, quantized)
                logger.info("[create_local_model] 4-bit quantization applied (nf4)")
            except ImportError:
                logger.warning("bitsandbytes not installed, 4-bit quantization DISABLED")
                logger.warning("Model will train in FP32 - install bitsandbytes for true QLoRA")
                return model, config
            model = prepare_model_for_kbit_training(model)
        elif load_in_8bit:
            logger.info("[create_local_model] Using 8-bit quantization")
            try:
                import bitsandbytes as bnb
                name_to_module = dict(model.named_modules())
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and not any(x in name for x in ['lora', 'head', 'embed']):
                        quantized = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            has_fp16_weights=False
                        )
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        if parent_name:
                            parent = name_to_module[parent_name]
                            setattr(parent, child_name, quantized)
                        else:
                            setattr(model, child_name, quantized)
                logger.info("[create_local_model] 8-bit quantization applied")
            except ImportError:
                logger.warning("bitsandbytes not installed, 8-bit quantization DISABLED")
                return model, config
            model = prepare_model_for_kbit_training(model)

    return model, config


def create_tokenizer(vocab_size: int = 32000):
    """
    Create tokenizer matching model vocab_size.
    Uses unified tokenizer module with SentencePiece if available, falls back to GPT2.
    """
    logger.info(f"[create_tokenizer] Creating tokenizer (vocab_size={vocab_size})")
    
    try:
        from models.tokenizer import get_tokenizer
        tokenizer = get_tokenizer("fusion", vocab_size=vocab_size)
        return tokenizer
    except Exception as e:
        logger.warning(f"Fusion tokenizer failed ({e}), falling back to GPT2")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def apply_lora(
    model,
    lora_rank: int = 64,
    lora_alpha: int = 16,
    target_modules: list = None,
):
    """
    应用 LoRA 适配器
    """
    if target_modules is None:
        # 目标模块（根据 FusionModel 的实际层名）
        # L-NEW-2 FIX: Remove "out_proj" (doesn't match model);
        # FusionModel follows LLaMA naming: q_proj/k_proj/v_proj/o_proj for attention,
        # gate_proj/up_proj/down_proj for MLP
        # S3 FIX: Include SBLAttention latent projections for proper LoRA coverage
        target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "latent_q_proj", "latent_k_proj", "latent_v_proj", "latent_out_proj",
            "v_to_hidden_proj",
        ]
    
    logger.info(f"[apply_lora] 应用 LoRA（rank={lora_rank}, alpha={lora_alpha}）")
    logger.info(f"[apply_lora] 目标模块：{target_modules}")
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train(args):
    """
    主训练函数
    """
    logger.info("=" * 60)
    logger.info("[train] 开始训练 Fusion 模型")
    logger.info(f"  模型大小：{args.model_size}")
    logger.info(f"  量化：{args.quantize}（4bit={args.load_in_4bit}, 8bit={args.load_in_8bit}）")
    logger.info(f"  LoRA：{args.use_lora}（rank={args.lora_rank}）")
    logger.info(f"  数据路径：{args.data_path}")
    logger.info("=" * 60)
    
    # 1. 加载 tokenizer
    tokenizer = create_tokenizer(vocab_size={
        "0.5B": 32000, "1.5B": 32000, "8B": 100000, "14B": 100000
    }.get(args.model_size, 32000))
    
    # S3 fix: sync vocab_size to actual tokenizer
    actual_vocab_size = len(tokenizer)
    
    # 2. 创建模型（本地随机初始化）
    model, config = create_local_model(
        model_size=args.model_size,
        quantize=args.quantize,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        vocab_size_override=actual_vocab_size,
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
        report_to=args.report_to,
        deepspeed=args.deepspeed_config if args.use_deepspeed else None,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
    )
    
    # 6. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 7. 开始训练
    logger.info("[train] 开始训练循环...")
    trainer.train()
    
    # 8. 保存模型
    logger.info(f"[train] 保存模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 保存 FusionConfig
    config_path = os.path.join(args.output_dir, "fusion_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info("[train] 训练完成！")


def main():
    parser = argparse.ArgumentParser(description="Fusion 模型 LoRA/QLoRA 微调")
    
    # 模型参数
    parser.add_argument("--model_size", type=str, default="1.5B",
                        choices=["0.5B", "1.5B", "8B", "14B"],
                        help="模型大小（0.5B/1.5B/8B/14B）")
    parser.add_argument("--local_model", action="store_true", default=True,
                        help="使用本地 FusionModel（默认，无需预训练权重）")
    parser.add_argument("--quantize", action="store_true",
                        help="是否使用量化（QLoRA）")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="4-bit 量化（NF4）")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="8-bit 量化")
    
    # LoRA 参数
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="是否使用 LoRA（默认开启）")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA 秩（rank）")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    
    # 训练参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据路径（JSON 格式）")
    parser.add_argument("--output_dir", type=str, default="./output/fusion-lora",
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
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="预热步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪")
    
    # 混合精度
    parser.add_argument("--fp16", action="store_true",
                        help="使用 FP16 混合精度")
    parser.add_argument("--bf16", action="store_true",
                        help="使用 BF16 混合精度（推荐）")
    
    # 日志和保存
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志打印间隔（步数）")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存检查点间隔（步数）")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="最多保存的检查点数")
    parser.add_argument("--report_to", type=str, default="none",
                        help="日志报告目标（none/tensorboard/wandb）")
    
    # DeepSpeed
    parser.add_argument("--use_deepspeed", action="store_true",
                        help="是否使用 DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="DeepSpeed 配置文件路径")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.quantize and not (args.load_in_4bit or args.load_in_8bit):
        raise ValueError("使用 --quantize 时必须指定 --load_in_4bit 或 --load_in_8bit")
    
    # 默认开启 BF16
    if not args.fp16 and not args.bf16:
        args.bf16 = True
    
    # 开始训练
    train(args)


if __name__ == "__main__":
    main()