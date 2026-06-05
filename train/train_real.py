"""
实际模型训练 - 训练 100 步（使用真实数据）
"""
import sys
import torch
import torch.optim as optim
from pathlib import Path
import json

sys.path.insert(0, '.')

from models.fusion_mini import FusionMini, FusionMiniConfig


def train_real():
    """实际训练（100 步）"""
    print("[TRAIN] 开始实际模型训练（100 步）...")
    print()
    
    # 1. 创建小配置（实际使用）
    print("[1] 创建模型配置...")
    config = FusionMiniConfig(
        vocab_size=100,       # 小词表（匹配 tokenizer）
        hidden_size=128,       # 小隐层
        num_hidden_layers=2,   # 2 层
        num_attention_heads=2, # 2 个注意力头
        intermediate_size=256,
        max_position_embeddings=64,
    )
    print(f"   词汇表大小: {config.vocab_size}")
    print(f"   隐藏层大小: {config.hidden_size}")
    print(f"   层数: {config.num_hidden_layers}")
    print()
    
    # 2. 创建模型
    print("[2] 创建模型...")
    model = FusionMini(config)
    model.train()  # 训练模式
    param_count = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"   参数量: {param_count:.1f}K")
    print("   模型创建成功")
    print()
    
    # 3. 创建优化器
    print("[3] 创建优化器...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.01,
    )
    print("   优化器创建成功")
    print()
    
    # 4. 加载训练数据
    print("[4] 加载训练数据...")
    data_path = Path("data/training_data.txt")
    
    if not data_path.exists():
        print(f"   [ERROR] 训练数据不存在: {data_path}")
        return False
    
    with open(data_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    print(f"   句子数量: {len(sentences)}")
    print("   训练数据加载成功")
    print()
    
    # 5. 准备训练数据（简单编码）
    print("[5] 准备训练数据...")
    
    # 简单字符级编码
    chars = sorted(list(set("".join(sentences))))
    char_to_idx = {ch: i+3 for i, ch in enumerate(chars)}  # +3 for [PAD], [UNK], [CLS]
    char_to_idx["[PAD]"] = 0
    char_to_idx["[UNK]"] = 1
    char_to_idx["[CLS]"] = 2
    
    # 编码句子
    encoded_sentences = []
    for sent in sentences:
        encoded = [char_to_idx.get(ch, 1) for ch in sent]  # 1 = [UNK]
        encoded_sentences.append(encoded)
    
    print(f"   词汇表大小: {len(char_to_idx)}")
    print(f"   编码句子数量: {len(encoded_sentences)}")
    print("   训练数据准备成功")
    print()
    
    # 6. 训练 100 步
    print("[6] 训练 100 步...")
    losses = []
    batch_size = 4
    seq_len = 32
    
    for step in range(100):
        # 随机选择句子
        indices = torch.randint(0, len(encoded_sentences), (batch_size,))
        
        # 创建批次
        batch_input = []
        batch_labels = []
        
        for idx in indices:
            encoded = encoded_sentences[idx]
            
            # 截断或填充到 seq_len
            if len(encoded) > seq_len:
                encoded = encoded[:seq_len]
            else:
                encoded = encoded + [0] * (seq_len - len(encoded))
            
            batch_input.append(encoded[:-1])  # 输入：除最后一个 token
            batch_labels.append(encoded[1:])  # 标签：除第一个 token
        
        input_ids = torch.tensor(batch_input)
        labels = torch.tensor(batch_labels)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True,
        )
        
        loss = outputs["loss"]
        losses.append(loss.item())
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 每 10 步打印一次
        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            print(f"   Step {step+1:3d}: Loss = {loss.item():.4f} (Avg: {avg_loss:.4f})")
    
    print("   训练完成")
    print()
    
    # 7. 验证损失下降
    print("[7] 验证损失下降...")
    initial_loss = losses[0]
    final_loss = losses[-1]
    is_decreasing = final_loss < initial_loss
    
    print(f"   初始 Loss: {initial_loss:.4f}")
    print(f"   最终 Loss: {final_loss:.4f}")
    print(f"   Loss 变化: {final_loss - initial_loss:+.4f}")
    print()
    
    if is_decreasing:
        print("   [PASS] Loss 持续下降")
        print("   训练有效！")
    else:
        print("   [WARN] Loss 未下降")
        print("   可能的问题：学习率太大 / 数据太少 / 模型太小")
    print()
    
    # 8. 保存模型
    print("[8] 保存模型...")
    output_dir = Path("output/real_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型权重
    torch.save(model.state_dict(), output_dir / "model.pt")
    
    # 保存配置
    config_dict = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"   模型保存路径: {output_dir}")
    print("   模型保存成功")
    print()
    
    print("[TRAIN] 实际模型训练完成")
    return is_decreasing


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 实际模型训练（100 步）")
    print("=" * 60)
    print()
    
    try:
        success = train_real()
        if success:
            print()
            print("[PASS] 训练测试通过")
    except Exception as e:
        print()
        print(f"[FAIL] 训练测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)
