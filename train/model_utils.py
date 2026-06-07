"""Shared model creation utilities for Fusion-LLM training scripts.

S4 FIX: Extract duplicated create_local_model() from lora_finetune.py and
full_finetune.py into a single source of truth.
"""

import torch
import logging

logger = logging.getLogger(__name__)

# Model size presets
MODEL_CONFIGS = {
    "0.5B": dict(vocab_size=32000, hidden_size=2048, num_hidden_layers=16,
                 num_attention_heads=16, num_key_value_heads=8, intermediate_size=5504),
    "1.5B": dict(vocab_size=32000, hidden_size=3072, num_hidden_layers=24,
                 num_attention_heads=24, num_key_value_heads=8, intermediate_size=8192),
    "8B": dict(vocab_size=100000, hidden_size=4096, num_hidden_layers=32,
               num_attention_heads=32, num_key_value_heads=8, intermediate_size=11008),
    "14B": dict(vocab_size=100000, hidden_size=5120, num_hidden_layers=40,
                num_attention_heads=40, num_key_value_heads=8, intermediate_size=13824),
}

COMMON_CONFIG = dict(
    block_size=512,
    latent_dim=64,
    window_size=2048,
    sbla_mode="hybrid",
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    enable_thinking_dial=True,
    num_thinking_depths=4,
)


def create_local_model(
    model_size: str = "8B",
    torch_dtype: torch.dtype = torch.bfloat16,
    vocab_size_override: int | None = None,
):
    """Create a locally-initialized FusionModel (no pretrained weights required).

    Args:
        model_size: One of "0.5B", "1.5B", "8B", "14B"
        torch_dtype: Model dtype (default bfloat16)
        vocab_size_override: Override vocab_size to match actual tokenizer

    Returns:
        Tuple of (FusionModel instance, FusionConfig) — matches train script wrapper API (N15 FIX)
    """
    from models.fusion_model import FusionConfig, FusionModel

    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model size: {model_size}, options: {list(MODEL_CONFIGS.keys())}")

    config_dict = MODEL_CONFIGS[model_size].copy()

    if vocab_size_override is not None:
        config_dict['vocab_size'] = vocab_size_override

    config = FusionConfig(**config_dict, **COMMON_CONFIG)

    logger.info(f"[create_local_model] Creating Fusion-{model_size} model")
    logger.info(f"  vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, "
                f"layers={config.num_hidden_layers}, heads={config.num_attention_heads}")

    model = FusionModel(config)

    if torch_dtype is not None:
        model = model.to(torch_dtype)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {param_count / 1e9:.2f}B")

    return model, config  # N15 FIX: Return (model, config) tuple for API consistency
