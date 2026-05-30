"""Debug attention_mask handling"""
import sys
sys.path.insert(0, ".")
import torch

print("[DEBUG] Testing attention_mask handling...")

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

batch_size, seq_len = 2, 32
input_ids = torch.randint(0, 10000, (batch_size, seq_len))

# Case 1: No attention_mask
hs = model.embeddings(input_ids)
hs = model.dropout(hs)
for i, layer in enumerate(model.layers):
    hs, _ = layer(hs)
hs = model.norm(hs)
logits1 = model.lm_head(hs)
print(f"No mask logits: min={logits1.min():.4f}, max={logits1.max():.4f}, has_nan: {torch.isnan(logits1).any()}")

# Case 2: attention_mask as 2D
mask_2d = torch.ones(batch_size, seq_len)
hs = model.embeddings(input_ids)
hs = model.dropout(hs)
for i, layer in enumerate(model.layers):
    hs, _ = layer(hs, attention_mask=mask_2d)
hs = model.norm(hs)
logits2 = model.lm_head(hs)
print(f"2D mask logits: min={logits2.min():.4f}, max={logits2.max():.4f}, has_nan: {torch.isnan(logits2).any()}")

# Case 3: What does _build_causal_mask return for seq_len=32?
device = hs.device
causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))
print(f"Causal mask shape: {causal_mask.shape}, has -inf: {(causal_mask == float('-inf')).any()}")
print(f"Causal mask diag: {torch.diag(causal_mask)[:5]}")

# Case 4: What is combined_mask after adding window_mask?
window_mask = torch.zeros(seq_len, seq_len, device=device)
combined_mask = (causal_mask + window_mask).unsqueeze(0).unsqueeze(0)
print(f"Combined mask: shape={combined_mask.shape}, min={combined_mask.min()}, max={combined_mask.max()}")

# Check if softmax produces NaN when there are rows full of -inf
attn_scores_test = torch.randn(2, 4, 32, 32)
attn_scores_test = attn_scores_test + combined_mask
print(f"Attn scores after mask: has -inf rows: {(attn_scores_test == float('-inf')).any(dim=-1).all()}")
attn_probs = torch.softmax(attn_scores_test, dim=-1)
print(f"Attn probs: has nan: {torch.isnan(attn_probs).any()}")

# Case 5: Forward pass with full model
outputs = model(input_ids=input_ids, attention_mask=mask_2d, return_dict=True)
print(f"Full model logits: min={outputs['logits'].min():.4f}, max={outputs['logits'].max():.4f}, has_nan: {torch.isnan(outputs['logits']).any()}")