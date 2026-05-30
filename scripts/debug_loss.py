"""Debug script for NaN loss"""
import sys
sys.path.insert(0, ".")
import torch

print("[DEBUG] Testing Fusion Model loss calculation...")

# Import directly
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

# Small test
batch_size, seq_len = 2, 32
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=True)

logits = outputs["logits"]
print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
print(f"Logits has inf: {torch.isinf(logits).any()}")
print(f"Logits has nan: {torch.isnan(logits).any()}")

# Check logits at last position
last_logits = logits[:, -1, :]
print(f"Last token logits: min={last_logits.min():.4f}, max={last_logits.max():.4f}")

# Try computing loss manually
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = input_ids[..., 1:].contiguous()
print(f"Shift logits stats: min={shift_logits.min():.4f}, max={shift_logits.max():.4f}")
print(f"Shift logits has inf: {torch.isinf(shift_logits).any()}")
print(f"Shift labels range: {shift_labels.min():.0f} - {shift_labels.max():.0f}")

loss_fct = torch.nn.CrossEntropyLoss()
try:
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    print(f"Manual loss: {loss.item():.4f}")
except Exception as e:
    print(f"Loss computation error: {e}")

# Check for any NaN in embeddings
embeds = model.embeddings(input_ids)
print(f"Embeddings: min={embeds.min():.4f}, max={embeds.max():.4f}, has_nan={torch.isnan(embeds).any()}")

# Test a simple forward without labels
outputs_no_labels = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
print(f"Forward no labels logits: min={outputs_no_labels['logits'].min():.4f}, max={outputs_no_labels['logits'].max():.4f}")