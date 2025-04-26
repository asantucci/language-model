import torch
import pytest
from model.rope import RoPE

@pytest.mark.parametrize("batch_size, heads, seq_len, embed_dim", [
    (2, 4, 8, 16),
    (1, 2, 32, 64),
    (3, 8, 16, 32),
])
def test_rope_shape(batch_size, heads, seq_len, embed_dim):
    x = torch.randn(batch_size, heads, seq_len, embed_dim)
    rope = RoPE(embed_dim=embed_dim)
    out = rope(x)
    assert out.shape == x.shape, "Output shape must match input shape."

def test_rope_no_nan_inf():
    batch_size, heads, seq_len, embed_dim = 2, 4, 16, 32
    x = torch.randn(batch_size, heads, seq_len, embed_dim)
    rope = RoPE(embed_dim=embed_dim)
    out = rope(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf values."

def test_rope_manual_small_input():
    """
    Check that for a simple case like seq_len=2, embed_dim=4,
    the cosine/sine operations act as expected.
    """
    rope = RoPE(embed_dim=4, base=10)
    x = torch.ones(1, 1, 2, 4)  # batch=1, heads=1, seq_len=2, embed_dim=4.
    out = rope(x)

    # After RoPE, output should still be finite and structured.
    assert out.shape == (1, 1, 2, 4)
    assert torch.isfinite(out).all()

    # Check if output changes across positions.
    diff = (out[..., 0, :] - out[..., 1, :]).abs().sum()
    assert diff > 0, "Rotary embeddings should differentiate positions."

def test_rope_cache_efficiency():
    """
    Verify that calling _build_cache with smaller seq_len doesn't rebuild.
    """
    rope = RoPE(embed_dim=16)
    rope._build_cache(seq_len=32)
    old_cos = rope.cos_cache
    rope._build_cache(seq_len=16)  # Should not rebuild.
    assert rope.cos_cache.data_ptr() == old_cos.data_ptr(), "Should reuse cache for smaller seq_len."

    # Now force rebuild.
    rope._build_cache(seq_len=64)
    assert rope.cos_cache.shape[-2] >= 64, "Cache should expand if needed."
