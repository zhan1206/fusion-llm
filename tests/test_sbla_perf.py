"""
SBLA Attention performance profiling tests.

Migrated from print-based to pytest convention (D16 audit fix).
Uses forward_with_qkv entry point (forward() not implemented on SBLAttention).
"""
import sys
import torch
import time
import pytest

sys.path.insert(0, '.')

from models.sbla_attention import SBLAttention


def _profile_sbla_impl(batch_size=1, seq_len=32, hidden_size=64, num_heads=2, window_size=16, num_runs=10):
    """Core profiling logic using forward_with_qkv."""
    attention = SBLAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        window_size=window_size,
        block_size=8,
        latent_dim=8,
        mode="pure_sbla",
    )
    attention.eval()

    device = next(attention.parameters()).device
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)

    q_proj = torch.nn.Linear(hidden_size, hidden_size).to(device)
    k_proj = torch.nn.Linear(hidden_size, hidden_size).to(device)
    v_proj = torch.nn.Linear(hidden_size, hidden_size).to(device)
    Q = q_proj(hidden_states).view(batch_size, seq_len, num_heads, hidden_size // num_heads).transpose(1, 2)
    K = k_proj(hidden_states).view(batch_size, seq_len, num_heads, hidden_size // num_heads).transpose(1, 2)
    V = v_proj(hidden_states).view(batch_size, seq_len, num_heads, hidden_size // num_heads).transpose(1, 2)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    # Warmup
    with torch.no_grad():
        out = attention.forward_with_qkv(Q, K, V, attention_mask=attention_mask)
    assert out is not None

    # Profile
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            out = attention.forward_with_qkv(Q, K, V, attention_mask=attention_mask)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - t0

    avg_ms = elapsed * 1000 / num_runs
    return avg_ms


def test_sbla_perf_forward():
    """Test that SBLA attention forward runs and completes in reasonable time."""
    avg_ms = _profile_sbla_impl(batch_size=1, seq_len=32, num_runs=10)
    assert avg_ms > 0
    assert avg_ms < 1000  # Should complete in under 1 second


def test_sbla_perf_longer_sequence():
    """Test SBLA with longer sequence."""
    avg_ms = _profile_sbla_impl(batch_size=1, seq_len=128, num_runs=5)
    assert avg_ms > 0
    assert avg_ms < 5000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sbla_perf_cuda():
    """Test SBLA performance on CUDA (if available)."""
    device = torch.device("cuda")
    attention = SBLAttention(hidden_size=64, num_heads=2, window_size=16, block_size=8, latent_dim=8, mode="pure_sbla")
    attention.to(device)
    attention.eval()

    batch_size, seq_len = 1, 64
    hidden_states = torch.randn(batch_size, seq_len, 64, device=device)
    q_proj = torch.nn.Linear(64, 64).to(device)
    k_proj = torch.nn.Linear(64, 64).to(device)
    v_proj = torch.nn.Linear(64, 64).to(device)
    Q = q_proj(hidden_states).view(batch_size, seq_len, 2, 32).transpose(1, 2)
    K = k_proj(hidden_states).view(batch_size, seq_len, 2, 32).transpose(1, 2)
    V = v_proj(hidden_states).view(batch_size, seq_len, 2, 32).transpose(1, 2)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    with torch.no_grad():
        out = attention.forward_with_qkv(Q, K, V, attention_mask=attention_mask)
    assert out is not None
    assert isinstance(out, tuple)
    assert out[0].shape == (batch_size, seq_len, 64)
