"""Debug attention step by step"""
import sys
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
import math

print("[DEBUG] Step-by-step attention debugging...")

from models.fusion_model import FusionConfig

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

# Manual attention computation
batch_size, seq_len = 2, 32
hidden_states = torch.randn(batch_size, seq_len, 256)
device = hidden_states.device

# Q/K/V
q_proj = torch.nn.Linear(256, 256, bias=False)
k_proj = torch.nn.Linear(256, 256, bias=False)
v_proj = torch.nn.Linear(256, 256, bias=False)

Q = q_proj(hidden_states).view(batch_size, seq_len, 4, 64).transpose(1, 2)
K = k_proj(hidden_states).view(batch_size, seq_len, 4, 64).transpose(1, 2)
V = v_proj(hidden_states).view(batch_size, seq_len, 4, 64).transpose(1, 2)

print(f"Q shape: {Q.shape}, has_nan: {torch.isnan(Q).any()}")
print(f"K shape: {K.shape}, has_nan: {torch.isnan(K).any()}")
print(f"V shape: {V.shape}, has_nan: {torch.isnan(V).any()}")

# Compute attention scores
attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(64)
print(f"Attn scores: min={attn_scores.min():.4f}, max={attn_scores.max():.4f}, has_nan: {torch.isnan(attn_scores).any()}")

# Check for -inf in scores
print(f"Scores has -inf: {torch.isinf(attn_scores).any()}")
print(f"Scores has inf: {torch.isinf(attn_scores).any()}")

# Check causal mask
causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))
print(f"Causal mask: min={causal_mask.min():.4f}, max={causal_mask.max():.4f}")

# Add causal mask
attn_scores_masked = attn_scores + causal_mask
print(f"Scores after mask: min={attn_scores_masked.min():.4f}, has_nan: {torch.isnan(attn_scores_masked).any()}")

# Softmax
attn_probs = F.softmax(attn_scores_masked, dim=-1)
print(f"Attn probs: min={attn_probs.min():.4f}, max={attn_probs.max():.4f}, has_nan: {torch.isnan(attn_probs).any()}")

# Context
context = torch.matmul(attn_probs, V)
print(f"Context: min={context.min():.4f}, max={context.max():.4f}, has_nan: {torch.isnan(context).any()}")

# Reshape
context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, 256)
print(f"Context reshaped: min={context.min():.4f}, max={context.max():.4f}, has_nan: {torch.isnan(context).any()}")

# Output projection
out_proj = torch.nn.Linear(256, 256, bias=False)
output = out_proj(context)
print(f"Output: min={output.min():.4f}, max={output.max():.4f}, has_nan: {torch.isnan(output).any()}")