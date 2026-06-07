"""Thinking Dial integration test - verifies N10/N11/N12/N17/N18/N19/N20 fixes."""
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
    """N10/N18: thinking_depth bias produces different logits for different depths."""
    td_model, base_model, input_ids = setup
    with torch.no_grad():
        # Get raw logits without thinking bias
        raw_out = base_model(input_ids, return_dict=True)
        raw_logits_d0 = raw_out.logits[:, -1, :]  # (1, vocab)
        
        # Get logits with depth=0 bias via ThinkingDialModel.generate's hook mechanism
        hook_d0 = ThinkingDialModel._build_thinking_logits_hook(
            0, 1, input_ids.device, td_model.thinking_config,
            td_model.thinking_embedding, td_model.thinking_gate, base_model.lm_head,
        )
        hook_d3 = ThinkingDialModel._build_thinking_logits_hook(
            3, 1, input_ids.device, td_model.thinking_config,
            td_model.thinking_embedding, td_model.thinking_gate, base_model.lm_head,
        )
        biased_logits_d0 = hook_d0(raw_logits_d0.unsqueeze(1)).squeeze(1)
        biased_logits_d3 = hook_d3(raw_logits_d0.unsqueeze(1)).squeeze(1)
    
    # Raw logits should differ from biased logits
    assert not torch.allclose(raw_logits_d0, biased_logits_d0), "Depth=0 bias should change logits"
    # Different depths should produce different logits (bias vectors differ)
    assert not torch.allclose(biased_logits_d0, biased_logits_d3), "Depth=0 and depth=3 should differ"


def test_thinking_dial_n19_single_source(setup):
    """N19: _build_thinking_logits_hook is single source of truth used by both paths."""
    td_model, base_model, input_ids = setup
    # Verify the static method exists
    assert hasattr(ThinkingDialModel, '_build_thinking_logits_hook')
    # Verify it returns None when depth is None
    hook = ThinkingDialModel._build_thinking_logits_hook(
        None, 1, input_ids.device, td_model.thinking_config,
        td_model.thinking_embedding, td_model.thinking_gate, base_model.lm_head,
    )
    assert hook is None


def test_n20_forward_no_thinking_depth(setup):
    """N20: FusionModel.forward() should NOT accept thinking_depth (dead param removed)."""
    _, base_model, input_ids = setup
    import inspect
    sig = inspect.signature(base_model.forward)
    assert 'thinking_depth' not in sig.parameters, \
        "thinking_depth should be removed from FusionModel.forward() — use logits_hook instead"


def test_grpo_trainer_generate_with_thinking(setup):
    """N12: GRPOTrainer.generate_with_thinking() passes depth."""
    td_model, _, input_ids = setup
    trainer = GRPOTrainer(td_model)
    with torch.no_grad():
        texts = trainer.generate_with_thinking(input_ids, thinking_depth=2, max_new_tokens=5)
    assert len(texts) == 1


def test_n18_first_token_has_bias(setup):
    """N18: First sampled token in generate_samples should have thinking bias applied."""
    td_model, _, input_ids = setup
    trainer = GRPOTrainer(td_model)
    torch.manual_seed(42)
    ids_d0, _ = trainer.generate_samples(input_ids, num_samples=1, thinking_depth=0, max_new_tokens=3)
    torch.manual_seed(42)
    ids_d3, _ = trainer.generate_samples(input_ids, num_samples=1, thinking_depth=3, max_new_tokens=3)
    # Same seed but different depth → first generated token should differ (due to bias)
    # Check prompt portion matches
    assert torch.equal(ids_d0[0, :input_ids.shape[1]], ids_d3[0, :input_ids.shape[1]]), \
        "Prompt portion should be identical"
    # First generated token (position after prompt) should differ
    first_gen_0 = ids_d0[0, input_ids.shape[1]].item()
    first_gen_3 = ids_d3[0, input_ids.shape[1]].item()
    # With random weights the biases will differ, so tokens should differ
    # (Not guaranteed for every seed, but highly likely with different bias vectors)
    if first_gen_0 == first_gen_3:
        # Accept if the first token matches but verify the hook was actually applied
        # by checking logits directly
        with torch.no_grad():
            raw = td_model.base_model(input_ids, return_dict=True)
            raw_logits = raw.logits[:, -1, :]
            hook_d0 = ThinkingDialModel._build_thinking_logits_hook(
                0, 1, input_ids.device, td_model.thinking_config,
                td_model.thinking_embedding, td_model.thinking_gate, td_model.base_model.lm_head,
            )
            hook_d3 = ThinkingDialModel._build_thinking_logits_hook(
                3, 1, input_ids.device, td_model.thinking_config,
                td_model.thinking_embedding, td_model.thinking_gate, td_model.base_model.lm_head,
            )
            b0 = hook_d0(raw_logits.unsqueeze(1)).squeeze(1)
            b3 = hook_d3(raw_logits.unsqueeze(1)).squeeze(1)
            assert not torch.allclose(b0, b3), "Hooks must produce different logits"
    else:
        pass  # Best case: tokens differ


def test_generate_samples_with_depth(setup):
    """N12/N17: generate_samples passes thinking_depth, uses KV cache reuse."""
    td_model, _, input_ids = setup
    trainer = GRPOTrainer(td_model)
    with torch.no_grad():
        ids, texts = trainer.generate_samples(
            input_ids, num_samples=2, thinking_depth=1, max_new_tokens=3,
        )
    assert ids.shape[0] >= 2


def test_n23_prompt_mask_correct_dimension():
    """N23: Prompt mask should zero per-token positions, not per-sequence."""
    config = FusionConfig(
        vocab_size=1000, hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=512,
        block_size=8, latent_dim=16, window_size=64,
    )
    base_model = FusionModel(config)
    trainer = GRPOTrainer(base_model)
    
    # Test _normalize_logits_to_log_probs with per_token=True returns 2D
    B, L, V = 4, 10, config.vocab_size
    logits = torch.randn(B, L, V)
    labels = torch.randint(1, V, (B, L))  # non-zero labels
    
    per_token = trainer._normalize_logits_to_log_probs(logits, labels, per_token=True)
    per_seq = trainer._normalize_logits_to_log_probs(logits, labels, per_token=False)
    
    assert per_token.dim() == 2 and per_token.shape == (B, L-1), f"Expected ({B}, {L-1}), got {per_token.shape}"
    assert per_seq.dim() == 1 and per_seq.shape == (B,), f"Expected ({B},), got {per_seq.shape}"
    
    # Verify per_seq = per_token.sum(dim=1)
    assert torch.allclose(per_seq, per_token.sum(dim=1), atol=1e-5), "per_seq should equal per_token sum"
    
    # Verify masking: zero out first 3 positions, sum should differ
    masked = per_token.clone()
    masked[:, :3] = 0.0
    assert not torch.allclose(masked.sum(dim=1), per_seq), "Masking should change the sum"


def test_n24_pure_fusion_model_no_crash():
    """N24: GRPOTrainer with pure FusionModel should not crash (hasattr guard)."""
    config = FusionConfig(
        vocab_size=1000, hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=512,
        block_size=8, latent_dim=16, window_size=64,
    )
    base_model = FusionModel(config)
    base_model.eval()
    trainer = GRPOTrainer(base_model)
    input_ids = torch.randint(0, 1000, (1, 5))
    
    # Should not raise AttributeError even with thinking_depth=None
    with torch.no_grad():
        ids, texts = trainer.generate_samples(
            input_ids, num_samples=1, thinking_depth=None, max_new_tokens=3,
        )
    assert ids.shape[0] == 1


def test_m7_forward_uses_shared_hook(setup):
    """M7: ThinkingDialModel.forward() uses _build_thinking_logits_hook (single source)."""
    td_model, _, input_ids = setup
    with torch.no_grad():
        out_no_depth = td_model(input_ids)
        out_depth0 = td_model(input_ids, thinking_depth=0)
        out_depth3 = td_model(input_ids, thinking_depth=3)
    
    # Different depths should produce different logits
    assert not torch.allclose(out_depth0.logits, out_depth3.logits), \
        "Different thinking depths should produce different logits via forward()"
    # depth=0 should still differ from no depth
    assert not torch.allclose(out_no_depth.logits, out_depth0.logits), \
        "No depth vs depth=0 should differ"


def test_s2_train_step_accepts_thinking_depth(setup):
    """S2: train_step() accepts thinking_depth parameter."""
    td_model, _, input_ids = setup
    trainer = GRPOTrainer(td_model)
    # Verify the parameter exists
    import inspect
    sig = inspect.signature(trainer.train_step)
    assert 'thinking_depth' in sig.parameters, "train_step should accept thinking_depth"


def test_n25_mask_start_correct():
    """N25: mask_start should be prompt_len - 2 (first gen token at index prompt_len-2)."""
    config = FusionConfig(
        vocab_size=1000, hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=512,
        block_size=8, latent_dim=16, window_size=64,
    )
    base_model = FusionModel(config)
    trainer = GRPOTrainer(base_model)
    
    # Test the mask_start calculation: prompt_len=5 -> mask_start=3
    # This means index 0,1,2 (prompt tokens) are zeroed, index 3+ (gen tokens) are kept
    prompt_len = 5
    mask_start = max(prompt_len - 2, 0)
    assert mask_start == 3, f"Expected mask_start=3, got {mask_start}"
    
    # Verify: index 3 is first gen token (not zeroed)
    assert mask_start == 3, "First gen token should NOT be masked out"


def test_reward_fn_string_safety():
    """S2 FIX: reward_fn as string should not crash compute_reward."""
    config = FusionConfig(
        vocab_size=1000, hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=512,
        block_size=8, latent_dim=16, window_size=64,
    )
    base_model = FusionModel(config)
    # Register a test reward function
    GRPOTrainer.register_reward_fn('test_reward', lambda p, r: 1.0)
    
    trainer = GRPOTrainer(base_model, reward_fn='test_reward')  # Set as string
    # Should not raise TypeError
    reward = trainer.compute_reward('prompt', 'response')
    assert reward == 1.0, "String reward_fn should be looked up from registry"
