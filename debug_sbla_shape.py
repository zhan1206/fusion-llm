"""调试 SBLAttention 形状问题"""
import torch
import sys
sys.path.insert(0, '.')

from models.sbla_attention import SBLAttention

# 创建 SBLAttention (带 GQA)
sbla = SBLAttention(
    hidden_size=256,
    num_heads=8,
    num_key_value_heads=4,  # GQA: 8 heads, 4 KV heads
    block_size=32,
    latent_dim=64,
    window_size=64,
)

print(f"hidden_size: {sbla.hidden_size}")
print(f"num_heads: {sbla.num_heads}")
print(f"num_key_value_heads: {sbla.num_key_value_heads}")
print(f"num_kv_groups: {sbla.num_kv_groups}")
print(f"head_dim: {sbla.head_dim}")
print(f"kv_head_dim: {sbla.kv_head_dim}")
print()

# 创建测试输入
batch_size = 2
seq_len = 64
Q = torch.randn(batch_size, sbla.num_heads, seq_len, sbla.head_dim)
K = torch.randn(batch_size, sbla.num_key_value_heads, seq_len, sbla.kv_head_dim)
V = torch.randn(batch_size, sbla.num_key_value_heads, seq_len, sbla.kv_head_dim)

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
print()

# 模拟 forward_with_qkv() 中的代码
num_heads = sbla.num_heads
head_dim = sbla.head_dim
kv_head_dim = sbla.kv_head_dim
num_kv_groups = sbla.num_kv_groups

# _repeat_kv
V_full = sbla._repeat_kv(V, num_kv_groups)
print(f"V_full shape after _repeat_kv: {V_full.shape}")
print(f"  Expected: ({batch_size}, {num_heads}, {seq_len}, {kv_head_dim})")
print()

# transpose + view
V_transposed = V_full.transpose(1, 2)
print(f"V_transposed shape: {V_transposed.shape}")
print(f"  Expected: ({batch_size}, {seq_len}, {num_heads}, {kv_head_dim})")
print()

# 尝试 view 到 (batch_size, seq_len, hidden_size)
try:
    hidden_states_approx = V_transposed.contiguous().view(batch_size, seq_len, sbla.hidden_size)
    print(f"hidden_states_approx shape: {hidden_states_approx.shape}")
    print(f"  Success!")
except RuntimeError as e:
    print(f"ERROR: {e}")
    print()
    print(f"Expected hidden_size = {sbla.hidden_size}")
    print(f"But num_heads * kv_head_dim = {num_heads} * {kv_head_dim} = {num_heads * kv_head_dim}")
    print(f"Mismatch! Need to project from {num_heads * kv_head_dim} to {sbla.hidden_size}")
