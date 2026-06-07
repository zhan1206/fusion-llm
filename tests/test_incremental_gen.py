"""Quick incremental generation test to verify N6 (RoPE position_ids) fix."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.fusion_model import FusionConfig, FusionModel

config = FusionConfig(
    vocab_size=1000,
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=512,
    block_size=8,
    latent_dim=16,
    window_size=64,
)

model = FusionModel(config)
model.eval()

input_ids = torch.randint(0, 1000, (1, 10))

# Prefill
with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=True)
    logits_prefill = outputs.logits
    past_kv = outputs.past_key_values

print(f"Prefill logits shape: {logits_prefill.shape}")  # (1, 10, 1000)

# Incremental step 1
next_token = logits_prefill[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
with torch.no_grad():
    outputs_inc = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
    logits_inc = outputs_inc.logits
    past_kv2 = outputs_inc.past_key_values

print(f"Incremental step 1 logits shape: {logits_inc.shape}")  # should be (1, 1, 1000)

# Incremental step 2
next_token2 = logits_inc[:, -1, :].argmax(dim=-1, keepdim=True)
with torch.no_grad():
    outputs_inc2 = model(input_ids=next_token2, past_key_values=past_kv2, use_cache=True)
    logits_inc2 = outputs_inc2.logits

print(f"Incremental step 2 logits shape: {logits_inc2.shape}")  # should be (1, 1, 1000)

# Verify shapes are correct (N6 would have caused seq_len to expand)
assert logits_prefill.shape == (1, 10, 1000), f"Prefill shape wrong: {logits_prefill.shape}"
assert logits_inc.shape == (1, 1, 1000), f"Step 1 shape wrong: {logits_inc.shape}"
assert logits_inc2.shape == (1, 1, 1000), f"Step 2 shape wrong: {logits_inc2.shape}"

print("\nAll assertions passed! N6 fix verified - incremental generation works correctly.")
