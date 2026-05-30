"""Quick unit test for FusionModel"""
import sys
sys.path.insert(0, ".")
import torch

print("[TEST] Testing Fusion Model...")
model_module = __import__("models.fusion_model", fromlist=["FusionModel", "FusionConfig"])

# 创建小型配置
config = model_module.FusionConfig(
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

# 创建模型
model = model_module.FusionModel(config)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model created with {param_count:,} parameters")

# 前向传播
batch_size, seq_len = 2, 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=input_ids,
    return_dict=True,
)

print(f"Loss={outputs['loss'].item():.4f}, Logits shape={outputs['logits'].shape}")
assert outputs["loss"] is not None, "Loss should not be None"
assert not torch.isnan(outputs["loss"]).item(), "Loss is NaN!"
print("[PASS] FusionModel working!")