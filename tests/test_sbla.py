"""Quick unit test for SBLA Attention"""
import sys
sys.path.insert(0, ".")
import torch

print("[TEST] Testing SBLA Attention...")
sbla = __import__("models.sbla_attention", fromlist=["SBLAttention"]).SBLAttention(
    hidden_size=128,
    num_heads=4,
    block_size=16,
    latent_dim=32,
    window_size=16,
    mode="pure_sbla",
)

batch_size, seq_len = 2, 48
hidden_states = torch.randn(batch_size, seq_len, 128)
attention_mask = torch.ones(batch_size, 1, 1, seq_len)

output = sbla.forward(hidden_states=hidden_states, attention_mask=attention_mask)
print(f"OK: shape={output.shape}, no NaN={not torch.isnan(output).any()}")
print("[PASS] SBLA Attention working!")