"""
Tests for FusionModel.

Migrated from print-based to pytest convention (D16 audit fix).
"""
import sys
import torch
import pytest

sys.path.insert(0, ".")

from models.fusion_model import FusionConfig, FusionModel


def test_fusion_model_creation():
    """Test that FusionModel can be created with a minimal config."""
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
    assert model is not None
    assert isinstance(model, torch.nn.Module)


def test_fusion_model_forward():
    """Test forward pass produces expected output shapes."""
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

    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs is not None
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)


def test_fusion_model_parameter_count():
    """Test parameter counting works."""
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
    param_count = sum(p.numel() for p in model.parameters())
    assert isinstance(param_count, int)
    assert param_count > 0
