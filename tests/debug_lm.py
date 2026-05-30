"""Debug norm and lm_head"""
import sys
sys.path.insert(0, ".")
import torch

print("[DEBUG] Testing norm and lm_head...")

from models.fusion_model import FusionModel, FusionConfig

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

# Simulate hidden states after all layers
batch_size, seq_len = 2, 32
hidden_states = torch.randn(batch_size, seq_len, 256)

# Final norm
norm_out = model.norm(hidden_states)
print(f"Final norm output: min={norm_out.min():.4f}, max={norm_out.max():.4f}, has_nan: {torch.isnan(norm_out).any()}")

# LM head
logits = model.lm_head(norm_out)
print(f"Logits: min={logits.min():.4f}, max={logits.max():.4f}, has_nan: {torch.isnan(logits).any()}")

# Now test with actual layers
input_ids = torch.randint(0, 10000, (batch_size, seq_len))
embed_out = model.embeddings(input_ids)
print(f"Embeddings: min={embed_out.min():.4f}, max={embed_out.max():.4f}, has_nan: {torch.isnan(embed_out).any()}")

# Forward through layers
hs = model.dropout(embed_out)
print(f"After dropout: min={hs.min():.4f}, max={hs.max():.4f}, has_nan: {torch.isnan(hs).any()}")

for i, layer in enumerate(model.layers):
    hs, _ = layer(hs)
    print(f"Layer {i} output: min={hs.min():.4f}, max={hs.max():.4f}, has_nan: {torch.isnan(hs).any()}")

hs = model.norm(hs)
print(f"After final norm: min={hs.min():.4f}, max={hs.max():.4f}, has_nan: {torch.isnan(hs).any()}")

logits = model.lm_head(hs)
print(f"Final logits: min={logits.min():.4f}, max={logits.max():.4f}, has_nan: {torch.isnan(logits).any()}")