"""
Tests for incremental generation (N6 RoPE position_ids fix).

Migrated from print-based to pytest convention (D16 audit fix).
"""
import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_model import FusionConfig, FusionModel


def test_incremental_gen_prefill():
    """Test prefill step produces valid logits and past_key_values."""
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

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)

    assert outputs.logits is not None
    assert outputs.logits.shape == (1, 10, config.vocab_size)
    assert outputs.past_key_values is not None


def test_incremental_gen_decode():
    """Test decode step with past_key_values produces output for 1 token."""
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

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        next_token = torch.randint(0, 1000, (1, 1))
        outputs2 = model(
            input_ids=next_token,
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )

    assert outputs2.logits is not None
    assert outputs2.logits.shape == (1, 1, config.vocab_size)


def test_incremental_gen_consistency():
    """Test that prefill+decode produces non-NaN logits with correct shapes."""
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

    torch.manual_seed(42)
    input_ids = torch.randint(0, 1000, (1, 5))

    with torch.no_grad():
        full_outputs = model(input_ids=input_ids, use_cache=False)
        prefill = model(input_ids=input_ids[:, :-1], use_cache=True)
        decode = model(
            input_ids=input_ids[:, -1:],
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )

    # Both should produce valid logits (not NaN) with correct shapes
    assert not torch.isnan(full_outputs.logits).any()
    assert not torch.isnan(decode.logits).any()
    assert full_outputs.logits.shape == (1, 5, config.vocab_size)
    assert decode.logits.shape == (1, 1, config.vocab_size)
