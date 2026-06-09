"""
GRPO 训练脚本

使用 Group Relative Policy Optimization (GRPO) 训练 FusionModel。
支持 Thinking Dial 推理深度控制 + 奖励函数引导。

使用方法：
    python train/train_grpo.py \
        --model_path "./output/fusion-mini" \
        --train_data "data/gsm8k_train.jsonl" \
        --output_dir "./output/fusion-grpo" \
        --thinking_depth 2

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Optional, List, Dict
from torch.utils.data import Dataset, DataLoader
import logging

import sys
sys.path.insert(0, '.')

from models.thinking_dial import GRPOTrainer, GRPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRPODataset(Dataset):
    """GRPO 训练数据集"""

    def __init__(self, data_path: str, max_length: int = 512):
        self.max_length = max_length
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        logger.info(f"[OK] 加载 GRPO 数据：{len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item.get("prompt", item.get("question", "")),
            "reference": item.get("reference", item.get("answer", "")),
        }


def load_fusion_model(model_path: str, device: str = "auto"):
    """加载 Fusion 模型（优先 FusionMini，回退 FusionModel）"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    config = None

    # Try FusionMini first
    try:
        from models.fusion_mini import FusionMini, FusionMiniConfig
        model = FusionMini._load_from_safetensors(model_path)
        config = model.config
        logger.info(f"[OK] 加载 FusionMini: {model_path}")
    except Exception:
        pass

    # Try FusionModel
    if model is None:
        try:
            from models.fusion_model import FusionModel, FusionConfig
            model = FusionModel.from_pretrained(model_path)
            config = model.config
            logger.info(f"[OK] 加载 FusionModel: {model_path}")
        except Exception:
            pass

    # Fallback to HF AutoModel
    if model is None:
        import warnings
        warnings.warn(
            "无法加载 FusionModel/FusionMini，回退到 AutoModelForCausalLM。\n"
            "Thinking Dial（推理深度控制）在此模式下不可用。\n"
            "建议确认模型路径指向有效的 Fusion 模型。",
            UserWarning
        )
        logger.warning("[WARNING] 回退到 AutoModelForCausalLM，Thinking Dial 将不可用！")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        config = model.config
        logger.info(f"[OK] 加载 AutoModel (fallback): {model_path}")

    model.to(device)
    return model, config, device


def main():
    parser = argparse.ArgumentParser(description="Fusion-LLM GRPO Training")

    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据 (.jsonl)")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cuda/cpu)")

    # GRPO 参数
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--kl_coef", type=float, default=0.05, help="KL 散度系数")
    parser.add_argument("--reward_fn", type=str, default="gsm8k", help="奖励函数名 (gsm8k/length/combined)")
    parser.add_argument("--thinking_depth", type=int, default=2, help="Thinking Dial 深度 (0-3)")
    parser.add_argument("--num_generations", type=int, default=4, help="每个 prompt 生成数")
    parser.add_argument("--save_steps", type=int, default=100, help="保存间隔步数")

    args = parser.parse_args()

    # 加载模型
    model, config, device = load_fusion_model(args.model_path, args.device)

    # GRPO 配置
    grpo_config = GRPOConfig(
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        reward_fn_name=args.reward_fn,
        num_generations=args.num_generations,
    )

    # 创建 GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
    )

    # 加载数据
    dataset = GRPODataset(args.train_data, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 训练循环
    logger.info(f"[GO] GRPO 训练开始")
    logger.info(f"   数据: {len(dataset)} 条")
    logger.info(f"   批次: {args.batch_size}")
    logger.info(f"   轮数: {args.num_epochs}")
    logger.info(f"   Thinking depth: {args.thinking_depth}")
    logger.info(f"   奖励函数: {args.reward_fn}")

    global_step = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):
        epoch_reward = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            prompts = batch["prompt"]
            references = batch["reference"]

            # GRPO 训练步
            result = trainer.train_step(
                prompts=prompts,
                references=references,
                thinking_depth=args.thinking_depth,
            )

            epoch_reward += result.get("mean_reward", 0.0)
            num_batches += 1
            global_step += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"   Epoch {epoch+1}, Step {global_step}, "
                    f"Reward: {result.get('mean_reward', 0.0):.4f}, "
                    f"KL: {result.get('kl_divergence', 0.0):.4f}"
                )

            # 保存检查点
            if global_step % args.save_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                if hasattr(model, 'save_pretrained'):
                    model.save_pretrained(ckpt_dir)
                else:
                    torch.save(model.state_dict(), ckpt_dir / "model.pt")
                logger.info(f"   [OK] 检查点保存至: {ckpt_dir}")

        avg_reward = epoch_reward / max(num_batches, 1)
        logger.info(f"   Epoch {epoch+1}/{args.num_epochs} 完成, Avg Reward: {avg_reward:.4f}")

    # 保存最终模型
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(final_dir)
    else:
        torch.save(model.state_dict(), final_dir / "model.pt")

    logger.info(f"[DONE] GRPO 训练完成! 模型保存至: {final_dir}")


if __name__ == "__main__":
    main()
