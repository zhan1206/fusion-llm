"""
Tests for SBLA Attention.

Migrated from print-based to pytest convention (D16 audit fix).
"""
import sys
import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, ".")

from models.sbla_attention import SBLAttention


def test_sbla_attention_creation():
    """Test SBLAttention can be created."""
    sbla = SBLAttention(
        hidden_size=64,
        num_heads=4,
        block_size=8,
        latent_dim=8,
        window_size=16,
        mode="pure_sbla",
    )
    assert sbla is not None
    assert isinstance(sbla, torch.nn.Module)


def test_sbla_attention_forward():
    """Test forward_with_qkv returns output tensor with correct shape."""
    sbla = SBLAttention(
        hidden_size=64,
        num_heads=4,
        block_size=8,
        latent_dim=8,
        window_size=16,
        mode="pure_sbla",
    )
    sbla.eval()

    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, 64)
    attention_mask = torch.ones(batch_size, seq_len)

    q_proj = torch.nn.Linear(64, 64)
    k_proj = torch.nn.Linear(64, 64)
    v_proj = torch.nn.Linear(64, 64)
    Q = q_proj(hidden_states).view(batch_size, seq_len, 4, 16).transpose(1, 2)
    K = k_proj(hidden_states).view(batch_size, seq_len, 4, 16).transpose(1, 2)
    V = v_proj(hidden_states).view(batch_size, seq_len, 4, 16).transpose(1, 2)

    with torch.no_grad():
        output = sbla.forward_with_qkv(Q, K, V, attention_mask=attention_mask)

    assert output is not None
    assert isinstance(output, tuple)
    assert output[0].shape == (batch_size, seq_len, 64)


def test_sbla_attention_forward_with_qkv():
    """Test forward_with_qkv path specifically with direct QKV input."""
    sbla = SBLAttention(
        hidden_size=64,
        num_heads=4,
        block_size=8,
        latent_dim=8,
        window_size=16,
        mode="pure_sbla",
    )
    sbla.eval()

    batch_size, seq_len = 2, 16
    Q = torch.randn(batch_size, 4, seq_len, 16)
    K = torch.randn(batch_size, 4, seq_len, 16)
    V = torch.randn(batch_size, 4, seq_len, 16)
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        output = sbla.forward_with_qkv(Q, K, V, attention_mask=attention_mask)

    assert output is not None
    assert isinstance(output, tuple)
    assert output[0].shape == (batch_size, seq_len, 64)
