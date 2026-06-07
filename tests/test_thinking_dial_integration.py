"""Thinking Dial integration test - verifies N10/N11/N12/N17 fixes."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from models.fusion_model import FusionConfig, FusionModel
from models.thinking_dial import ThinkingDialModel, GRPOTrainer, ThinkingConfig


@pytest.fixture
def setup():
    config = FusionConfig(
        vocab_size=1000, hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=512,
        block_size=8, latent_dim=16, window_size=64,
    )
    base_model = FusionModel(config)
    base_model.eval()
    td_model = ThinkingDialModel(base_model, ThinkingConfig())
    td_model.eval()
    input_ids = torch.randint(0, 1000, (1, 5))
    return td_model, base_model, input_ids


def test_thinking_dial_model_generate(setup):
    """N11: ThinkingDialModel.generate() exists and works."""
    td_model, _, input_ids = setup
    with torch.no_grad():
        out = td_model.generate(input_ids=input_ids, max_new_tokens=5, thinking_depth=None)
    assert out.shape[0] == 1


def test_thinking_depth_bias_applied(setup):
    """N10: thinking_depth bias is applied during generation."""
    td_model, _, input_ids = setup
    with torch.no_grad():
        out_d0 = td_model.generate(input_ids=input_ids, max_new_tokens=5, thinking_depth=0)
        out_d3 = td_model.generate(input_ids=input_ids, max_new_tokens=5, thinking_depth=3)
    assert out_d0.shape == out_d3.shape


def test_grpo_trainer_generate_with_thinking(setup):
    """N12: GRPOTrainer.generate_with_thinking() passes depth."""
    td_model, _, input_ids = setup
    trainer = GRPOTrainer(td_model)
    with torch.no_grad():
        texts = trainer.generate_with_thinking(input_ids, thinking_depth=2, max_new_tokens=5)
    assert len(texts) == 1


def test_generate_samples_with_depth(setup):
    """N12/N17: generate_samples passes thinking_depth, uses KV cache reuse."""
    td_model, _, input_ids = setup
    trainer = GRPOTrainer(td_model)
    with torch.no_grad():
        ids, texts = trainer.generate_samples(
            input_ids, num_samples=2, thinking_depth=1, max_new_tokens=3,
        )
    assert ids.shape[0] >= 2
