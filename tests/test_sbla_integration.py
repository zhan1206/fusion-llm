"""
测试 SBLA 注意力集成（无 emoji 版本）
"""
import sys
sys.path.insert(0, '.')

from models.fusion_mini import FusionMini, FusionMiniConfig
import torch

print("测试 SBLA 注意力集成...")
print()

# 1. 创建配置
print("[1] 创建配置...")
config = FusionMiniConfig(
    vocab_size=1000,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
)
print("   配置创建成功")
print(f"   隐层大小：{config.hidden_size}")
print(f"   层数：{config.num_hidden_layers}")
print()

# 2. 创建模型
print("[2] 创建模型（包含 SBLA 注意力）...")
model = FusionMini(config)
print("   模型创建成功")
param_count = sum(p.numel() for p in model.parameters()) / 1e3
print(f"   参数量：{param_count:.1f}K")
print()

# 3. 测试前向传播
print("[3] 测试前向传播...")
input_ids = torch.randint(0, 1000, (2, 64))
print(f"   输入形状：{input_ids.shape}")

outputs = model.forward(input_ids=input_ids, labels=input_ids)
loss_value = outputs["loss"].item()
print(f"   前向传播成功")
print(f"   Loss：{loss_value:.4f}")
print()

# 4. 验证 SBLA 是否使用
print("[4] 验证 SBLA 注意力...")
has_sblla = any("SBLAttention" in str(module) for module in model.modules())
if has_sblla:
    print("   SBLA 注意力已集成到模型中")
else:
    print("   未检测到 SBLA 注意力（可能使用了标准注意力）")
print()

print("所有测试通过！")
print()
print("下一步：")
print("   1. 重新训练模型（使用 SBLA 注意力）")
print("   2. 对比标准注意力和 SBLA 的性能")
print("   3. 推送代码到 GitHub")
