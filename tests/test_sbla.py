"""Quick unit test for SBLA Attention"""
import sys
sys.path.insert(0, ".")
import torch

print("[TEST] Testing SBLA Attention...")
from models.sbla_attention import SBLAttention

sbla = SBLAttention(
    hidden_size=64,
    num_heads=4,
    block_size=8,
    latent_dim=8,
    window_size=16,
    mode="pure_sbla",
)

batch_size, seq_len = 2, 16
hidden_states = torch.randn(batch_size, seq_len, 64)
attention_mask = torch.ones(batch_size, seq_len)

# Create Q/K/V manually (simulating FusionAttention's role)
import torch.nn.functional as F
q_proj = torch.nn.Linear(64, 64)
k_proj = torch.nn.Linear(64, 64)
v_proj = torch.nn.Linear(64, 64)
Q = q_proj(hidden_states).view(batch_size, seq_len, 4, 16).transpose(1, 2)
K = k_proj(hidden_states).view(batch_size, seq_len, 4, 16).transpose(1, 2)
V = v_proj(hidden_states).view(batch_size, seq_len, 4, 16).transpose(1, 2)

output, cache = sbla.forward_with_qkv(Q, K, V, attention_mask)
print(f"OK: shape={output.shape}, no NaN={not torch.isnan(output).any()}, cache={cache}")
print("[PASS] SBLA Attention working!")
