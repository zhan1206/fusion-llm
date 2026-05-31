"""
Fusion Mini 训练脚本（可运行版本）

训练 fusion_mini 模型（极简版本，用于验证完整流程）

使用方法：
    # 1. 准备示例数据
    python tests/create_mini_data.py
    
    # 2. 训练模型
    python train/train_mini.py \
        --data_path data/mini_data.json \
        --output_dir output/mini_model \
        --num_epochs 3 \
        --batch_size 2 \
        --learning_rate 5e-4

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.fusion_mini import FusionMini, FusionMiniConfig
from models.tokenizer import get_tokenizer


def _try_get_tokenizer():
    """Try to load Fusion tokenizer, return None on failure."""
    try:
        tok = get_tokenizer("fusion")
        if tok is not None:
            print(f"   [Tokenizer] Loaded Fusion tokenizer, vocab_size={len(tok)}")
        return tok
    except Exception:
        return None


class MiniDataset(Dataset):
    """
    极简数据集
    
    用于训练 Fusion Mini 模型
    """
    
    def __init__(self, data_path: str, tokenizer=None, max_length: int = 128):
        """
        初始化数据集
        
        参数：
            data_path: 数据文件路径（JSON 格式）
            tokenizer: 分词器（如果没有，使用字符级）
            max_length: 最大序列长度
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"[数据] 加载数据集：{self.data_path}")
        print(f"   样本数：{len(self.data)}")
        
        # 预先构建字符索引（字符级编码）
        if self.tokenizer is None:
            # 收集所有字符
            all_chars = set()
            for item in self.data:
                text = f"{item['prompt']} {item['response']}"
                all_chars.update(list(text))
            
            # 创建字符到索引的映射
            self.char_to_idx = {c: i+4 for i, c in enumerate(sorted(all_chars))}
            print(f"   字符表大小：{len(self.char_to_idx)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建文本
        text = f"{item['prompt']} {item['response']}"
        
        # 编码（简化：使用字符级编码）
        if self.tokenizer is None:
            # 使用预先构建的字符索引
            chars = list(text)
            
            # 转换
            input_ids = [self.char_to_idx.get(c, 0) for c in chars[:self.max_length]]
            
            # 填充
            if len(input_ids) < self.max_length:
                input_ids = input_ids + [0] * (self.max_length - len(input_ids))
            else:
                input_ids = input_ids[:self.max_length]
            
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        else:
            # 使用 tokenizer
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != 0).long(),
            "labels": input_ids.clone(),
        }


def train_mini_model(
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-4,
    hidden_size: int = 128,
    num_hidden_layers: int = 4,
    max_length: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    训练 Fusion Mini 模型
    
    参数：
        data_path: 数据文件路径
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        hidden_size: 隐层大小
        num_hidden_layers: 层数
        max_length: 最大序列长度
        device: 设备
    """
    print("=" * 60)
    print("Fusion Mini 训练脚本")
    print("=" * 60)
    
    print(f"\n[配置] 训练配置：")
    print(f"   数据文件：{data_path}")
    print(f"   输出目录：{output_dir}")
    print(f"   训练轮数：{num_epochs}")
    print(f"   批次大小：{batch_size}")
    print(f"   学习率：{learning_rate}")
    print(f"   隐层大小：{hidden_size}")
    print(f"   层数：{num_hidden_layers}")
    print(f"   设备：{device}")
    
    # 1. 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 2. 加载数据集
    print(f"\n[数据] 加载数据集...")
    tok = _try_get_tokenizer()
    if tok is not None:
        vocab_size = len(tok)
    else:
        vocab_size = 1000
    dataset = MiniDataset(
        data_path=data_path,
        tokenizer=tok,  # Use fusion tokenizer if available, else char-level
        max_length=max_length,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    # 3. 创建模型配置
    print(f"\n[模型] 创建模型...")
    config = FusionMiniConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=max_length,
    )
    
    # If char-level tokenizer, adjust vocab_size from data
    if tok is None and hasattr(dataset, 'char_to_idx'):
        config.vocab_size = len(dataset.char_to_idx) + 10
    
    print(f"   词表大小：{config.vocab_size}")
    print(f"   隐层大小：{config.hidden_size}")
    print(f"   层数：{config.num_hidden_layers}")
    
    # 4. 创建模型
    model = FusionMini(config)
    model = model.to(device)
    
    print(f"\n[完成] 模型创建成功")
    print(f"   参数量：{sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
    
    # 5. 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # 6. 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(dataloader),
    )
    
    # 7. 训练循环
    print(f"\n[训练] 开始训练...")
    
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # 移动数据到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            
            loss = outputs["loss"]
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
            })
        
        # Epoch 总结
        avg_loss = total_loss / num_batches
        print(f"\n[完成] Epoch {epoch+1} 完成")
        print(f"   平均损失：{avg_loss:.4f}")
        
        # 保存检查点
        checkpoint_path = output_path / f"checkpoint-epoch-{epoch+1}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": config.to_dict(),
        }, checkpoint_path)
        
        print(f"   检查点已保存：{checkpoint_path}")
    
    # 8. 保存最终模型
    final_model_path = output_path / "final_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, final_model_path)
    
    # 9. 保存配置文件
    config_path = output_path / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
    
    print(f"\n[完成] 训练完成！")
    print(f"   最终模型：{final_model_path}")
    print(f"   配置文件：{config_path}")
    
    # 10. 测试生成
    print(f"\n[测试] 测试生成...")
    
    model.eval()
    
    test_prompt = "解释人工智能"
    print(f"   测试提示词：{test_prompt}")
    
    # 编码（简化）
    test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device)  # 模拟输入
    
    with torch.no_grad():
        generated = model.generate(
            input_ids=test_input,
            max_new_tokens=20,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
        )
    
    print(f"   生成形状：{generated.shape}")
    print(f"   （注：这是随机生成，因为使用字符级编码）")
    
    print(f"\n[下一步]")
    print(f"   1. 使用真实分词器（如 SentencePiece）")
    print(f"   2. 增加数据量和训练轮数")
    print(f"   3. 实现 SBLA 注意力和 Thinking Dial")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="Fusion Mini 训练脚本"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/mini_data.json",
        help="训练数据文件路径（JSON 格式）",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/mini_model",
        help="输出目录",
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="批次大小",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="学习率",
    )
    
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="隐层大小",
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Transformer 层数",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="最大序列长度",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备（cuda/cpu）",
    )
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not Path(args.data_path).exists():
        print(f"[错误] 数据文件不存在：{args.data_path}")
        print(f"   请先运行：python tests/create_mini_data.py")
        return
    
    # 训练模型
    model, config = train_mini_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        max_length=args.max_length,
        device=args.device,
    )
    
    print(f"\n[完成] 训练完成！")


if __name__ == "__main__":
    main()
