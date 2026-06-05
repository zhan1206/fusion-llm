"""
图形版模型可视化工具 - 使用 matplotlib 绘制注意力热力图、损失曲线、模型架构图
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

try:
    import matplotlib.pyplot as plt
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib 未安装，将使用文本模式可视化")
    print("   安装命令: pip install matplotlib")


def visualize_attention_graphical(attention_weights, head_idx=0, max_len=32, output_path=None):
    """
    图形化注意力可视化（需要 matplotlib）
    
    Args:
        attention_weights: 注意力权重，形状为 (batch, num_heads, seq_len, seq_len)
        head_idx: 要可视化的注意力头索引
        max_len: 最大可视化长度
        output_path: 输出路径（如果提供，则保存为文件）
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib 未安装，跳过图形化注意力可视化")
        return
    
    print("[VISUALIZE] 图形化注意力可视化...")
    
    # 获取指定头的注意力权重
    attn = attention_weights[0, head_idx, :max_len, :max_len].cpu().numpy()  # (seq_len, seq_len)
    
    # 创建热力图
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(f'Attention Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=100)
        print(f"   注意力热力图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()
    print("   图形化注意力可视化完成")
    print()


def visualize_loss_curve_graphical(losses, window=10, output_path=None):
    """
    图形化损失曲线可视化（需要 matplotlib）
    
    Args:
        losses: 损失值列表
        window: 平滑窗口大小
        output_path: 输出路径（如果提供，则保存为文件）
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib 未安装，跳过图形化损失曲线可视化")
        return
    
    print("[VISUALIZE] 图形化损失曲线可视化...")
    
    if len(losses) < 2:
        print("   损失点太少，无法可视化")
        return
    
    # 平滑损失（移动平均）
    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window // 2)
        end = min(len(losses), i + window // 2 + 1)
        smoothed.append(sum(losses[start:end]) / (end - start))
    
    # 创建损失曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', alpha=0.3, label='Original')
    plt.plot(range(1, len(smoothed) + 1), smoothed, 'r-', label=f'Smoothed (window={window})')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=100)
        print(f"   损失曲线图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()
    print("   图形化损失曲线可视化完成")
    print()


def visualize_model_architecture_graphical(model, output_path=None):
    """
    图形化模型架构可视化（需要 matplotlib）
    
    Args:
        model: PyTorch 模型
        output_path: 输出路径（如果提供，则保存为文件）
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib 未安装，跳过图形化模型架构可视化")
        return
    
    print("[VISUALIZE] 图形化模型架构可视化...")
    
    # 获取所有模块
    modules = []
    params = []
    for name, module in model.named_modules():
        if name == "":
            continue
        modules.append(name)
        params.append(sum(p.numel() for p in module.parameters()))
    
    # 创建条形图
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(modules)), params, color='skyblue')
    plt.yticks(range(len(modules)), modules, fontsize=8)
    plt.xlabel('Parameters')
    plt.title('Model Architecture (Parameters per Module)')
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=100)
        print(f"   模型架构图已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()
    print("   图形化模型架构可视化完成")
    print()


def save_visualization_report_graphical(model, attention_weights, losses, output_dir):
    """
    保存图形化可视化报告到文件
    
    Args:
        model: PyTorch 模型
        attention_weights: 注意力权重
        losses: 损失值列表
        output_dir: 输出目录
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib 未安装，无法生成图形化报告")
        return
    
    print("[VISUALIZE] 保存图形化可视化报告...")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 注意力热力图
    if attention_weights is not None:
        attention_path = output_dir / "attention_heatmap.png"
        visualize_attention_graphical(
            attention_weights,
            head_idx=0,
            max_len=min(32, attention_weights.shape[2]),
            output_path=str(attention_path)
        )
    
    # 2. 损失曲线图
    if losses is not None and len(losses) > 1:
        loss_path = output_dir / "loss_curve.png"
        visualize_loss_curve_graphical(
            losses,
            window=10,
            output_path=str(loss_path)
        )
    
    # 3. 模型架构图
    architecture_path = output_dir / "model_architecture.png"
    visualize_model_architecture_graphical(
        model,
        output_path=str(architecture_path)
    )
    
    print(f"   图形化报告已保存到: {output_dir}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Fusion-LLM 图形化模型可视化工具测试")
    print("=" * 60)
    print()
    
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib 未安装，无法运行图形化测试")
        print("   安装命令: pip install matplotlib")
        sys.exit(1)
    
    # 1. 测试注意力热力图
    print("[1] 测试图形化注意力热力图...")
    attention_weights = torch.rand(1, 2, 8, 8)  # (batch, num_heads, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    visualize_attention_graphical(
        attention_weights,
        head_idx=0,
        max_len=8,
        output_path="output/attention_heatmap_test.png"
    )
    print()
    
    # 2. 测试损失曲线图
    print("[2] 测试图形化损失曲线图...")
    losses = [5.0, 4.5, 4.0, 3.5, 3.0, 2.8, 2.5, 2.3, 2.1, 2.0]
    
    visualize_loss_curve_graphical(
        losses,
        window=3,
        output_path="output/loss_curve_test.png"
    )
    print()
    
    # 3. 测试模型架构图
    print("[3] 测试图形化模型架构图...")
    from models.fusion_mini import FusionMini, FusionMiniConfig
    
    config = FusionMiniConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
    )
    model = FusionMini(config)
    
    visualize_model_architecture_graphical(
        model,
        output_path="output/model_architecture_test.png"
    )
    print()
    
    # 4. 保存完整报告
    print("[4] 保存图形化可视化报告...")
    save_visualization_report_graphical(
        model=model,
        attention_weights=attention_weights,
        losses=losses,
        output_dir="output/visualization_report_graphical"
    )
    print()
    
    print("[PASS] 图形化模型可视化工具测试通过")
    sys.exit(0)
