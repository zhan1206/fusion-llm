"""
模型可视化工具 - 注意力可视化、损失曲线可视化、模型架构可视化
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, '.')


def visualize_attention_text(attention_weights, head_idx=0, max_len=32):
    """
    文本化注意力可视化（不需要 matplotlib）
    
    Args:
        attention_weights: 注意力权重，形状为 (batch, num_heads, seq_len, seq_len)
        head_idx: 要可视化的注意力头索引
        max_len: 最大可视化长度
    """
    print("[VISUALIZE] 注意力可视化（文本模式）...")
    
    # 获取指定头的注意力权重
    attn = attention_weights[0, head_idx, :max_len, :max_len]  # (seq_len, seq_len)
    
    print(f"   注意力头: {head_idx}")
    print(f"   序列长度: {attn.shape[0]}")
    print()
    print("   注意力热力图（文本模式）:")
    print("   " + "-" * 34)
    
    for i in range(attn.shape[0]):
        row = "   |"
        for j in range(attn.shape[1]):
            value = attn[i, j].item()
            if value > 0.5:
                row += "##"  # ASCII: high attention
            elif value > 0.1:
                row += "=="  # ASCII: medium attention
            elif value > 0.01:
                row += "--"  # ASCII: low attention
            else:
                row += "  "   # ASCII: very low attention
        row += "|"
        print(row)
    
    print("   " + "-" * 34)
    print("   ## > 0.5   == > 0.1   -- > 0.01")
    print()


def visualize_loss_curve_text(losses, window=10):
    """
    文本化损失曲线可视化（不需要 matplotlib）
    
    Args:
        losses: 损失值列表
        window: 平滑窗口大小
    """
    print("[VISUALIZE] 损失曲线可视化（文本模式）...")
    
    if len(losses) < 2:
        print("   损失点太少，无法可视化")
        return
    
    # 平滑损失（移动平均）
    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window // 2)
        end = min(len(losses), i + window // 2 + 1)
        smoothed.append(sum(losses[start:end]) / (end - start))
    
    # 归一化到 0-50（用于可视化）
    min_loss = min(smoothed)
    max_loss = max(smoothed)
    if max_loss - min_loss < 1e-6:
        print("   损失变化太小，无法可视化")
        return
    
    normalized = [(x - min_loss) / (max_loss - min_loss) * 50 for x in smoothed]
    
    print(f"   损失范围: {min_loss:.4f} - {max_loss:.4f}")
    print(f"   平滑窗口: {window}")
    print()
    print("   损失曲线（文本模式）:")
    print("   Loss ^")
    print("        |")
    
    for i in range(50, -1, -1):
        row = "   "
        for j in range(len(normalized)):
            if abs(normalized[j] - i) < 0.5:
                row += "*"
            else:
                row += " "
        print(row)
    
    print("   " + "-" * len(losses))
    print("        Step ->")
    print()


def visualize_model_architecture_text(model):
    """
    文本化模型架构可视化
    
    Args:
        model: PyTorch 模型
    """
    print("[VISUALIZE] 模型架构可视化（文本模式）...")
    print()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")
    print()
    
    # 打印模型结构
    print("   模型架构:")
    print("   " + "=" * 56)
    
    for name, module in model.named_modules():
        if name == "":
            continue
        
        # 缩进
        depth = name.count('.')
        indent = "    " * (depth + 1)
        
        # 模块信息
        module_type = type(module).__name__
        
        # 参数量
        params = sum(p.numel() for p in module.parameters())
        
        # 输出
        if params > 0:
            print(f"{indent}{name}: {module_type} ({params:,} params)")
        else:
            print(f"{indent}{name}: {module_type}")
    
    print("   " + "=" * 56)
    print()


def save_visualization_report(model, attention_weights, losses, output_path):
    """
    保存可视化报告到文件
    
    Args:
        model: PyTorch 模型
        attention_weights: 注意力权重
        losses: 损失值列表
        output_path: 输出路径
    """
    print("[VISUALIZE] 保存可视化报告...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        # 重定向 print 到文件
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            print("=" * 60)
            print("Fusion-LLM 可视化报告")
            print("=" * 60)
            print()
        
            # 模型架构
            visualize_model_architecture_text(model)
        
            # 注意力可视化（如果有）
            if attention_weights is not None:
                visualize_attention_text(attention_weights, head_idx=0)
        
            # 损失曲线（如果有）
            if losses is not None and len(losses) > 1:
                visualize_loss_curve_text(losses, window=10)
        
            print()
            print("=" * 60)
            print("报告结束")
            print("=" * 60)
        
        finally:
            # 恢复 stdout
            sys.stdout = original_stdout
    
    print(f"   报告已保存到: {output_path}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 模型可视化工具测试")
    print("=" * 60)
    print()
    
    # 1. 测试模型架构可视化
    print("[1] 测试模型架构可视化...")
    from models.fusion_mini import FusionMini, FusionMiniConfig
    
    config = FusionMiniConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
    )
    model = FusionMini(config)
    
    visualize_model_architecture_text(model)
    print()
    
    # 2. 测试损失曲线可视化
    print("[2] 测试损失曲线可视化...")
    losses = [5.0, 4.5, 4.0, 3.5, 3.0, 2.8, 2.5, 2.3, 2.1, 2.0]
    visualize_loss_curve_text(losses, window=3)
    print()
    
    # 3. 测试注意力可视化（模拟）
    print("[3] 测试注意力可视化（模拟）...")
    attention_weights = torch.rand(1, 2, 8, 8)  # (batch, num_heads, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    visualize_attention_text(attention_weights, head_idx=0, max_len=8)
    print()
    
    # 4. 保存报告
    print("[4] 保存可视化报告...")
    output_path = Path("output/visualization_report.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_visualization_report(
        model=model,
        attention_weights=attention_weights,
        losses=losses,
        output_path=output_path,
    )
    print()
    
    print("[PASS] 模型可视化工具测试通过")
    sys.exit(0)
