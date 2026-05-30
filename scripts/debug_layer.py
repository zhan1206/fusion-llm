"""Debug layer by layer"""
import sys
sys.path.insert(0, ".")
import torch

print("[DEBUG] Testing layer by layer...")

from models.fusion_model import FusionModel, FusionConfig, FusionAttention

config = FusionConfig(
    vocab_size=10000,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=512,
    block_size=64,
    latent_dim=16,
    sbla_mode="pure_sbla",
    max_position_embeddings=256,
)

model = FusionModel(config)
model.eval()

# Get the attention layer
attn = model.layers[0].attention

batch_size, seq_len = 2, 32
hidden_states = torch.randn(batch_size, seq_len, 256)

# Test attention
attn_out = attn.forward(hidden_states)
print(f"Attention output: min={attn_out.min():.4f}, max={attn_out.max():.4f}, has_nan: {torch.isnan(attn_out).any()}")

# Test RMSNorm
norm = model.layers[0].input_layernorm
norm_out = norm.forward(hidden_states)
print(f"RMSNorm output: min={norm_out.min():.4f}, max={norm_out.max():.4f}, has_nan: {torch.isnan(norm_out).any()}")

# Test FFN
ffn = model.layers[0]
residual = hidden_states
norm1_out = ffn.input_layernorm(hidden_states)
attn_out = ffn.attention(norm1_out)
after_attn = residual + attn_out
print(f"After attention residual: min={after_attn.min():.4f}, max={after_attn.max():.4f}, has_nan: {torch.isnan(after_attn).any()}")

norm2_out = ffn.post_attention_layernorm(after_attn)
print(f"Post-attention norm: min={norm2_out.min():.4f}, max={norm2_out.max():.4f}, has_nan: {torch.isnan(norm2_out).any()}")

gate = torch.nn.functional.silu(ffn.gate_proj(norm2_out))
up = ffn.up_proj(norm2_out)
print(f"Gate: min={gate.min():.4f}, max={gate.max():.4f}, has_nan: {torch.isnan(gate).any()}")
print(f"Up: min={up.min():.4f}, max={up.max():.4f}, has_nan: {torch.isnan(up).any()}")

gate_up = gate * up
print(f"Gate*Up: min={gate_up.min():.4f}, max={gate_up.max():.4f}, has_nan: {torch.isnan(gate_up).any()}")

ffn_out = ffn.down_proj(gate_up)
print(f"FFN output: min={ffn_out.min():.4f}, max={ffn_out.max():.4f}, has_nan: {torch.isnan(ffn_out).any()}")

final = after_attn + ffn_out
print(f"Final layer output: min={final.min():.4f}, max={final.max():.4f}, has_nan: {torch.isnan(final).any()}")