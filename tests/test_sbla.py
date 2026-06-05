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

output, cache = sbla(hidden_states=hidden_states, attention_mask=attention_mask)
print(f"OK: shape={output.shape}, no NaN={not torch.isnan(output).any()}, cache={cache}")
print("[PASS] SBLA Attention working!")
