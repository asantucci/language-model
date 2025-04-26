import torch
import pytest
from model.kv_cache import KVCache

@pytest.fixture
def kv_cache():
    """Fixture to create a KVCache with 2 layers."""
    return KVCache(num_layers=2)

@pytest.mark.parametrize("layer_idx", [0, 1])
def test_initial_cache_is_empty(kv_cache, layer_idx):
    """Test that the cache is empty upon initialization."""
    assert kv_cache.get_cache_length(layer_idx) == 0

@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim",
    [
        (2, 4, 3, 8),
        (1, 8, 5, 16),
    ]
)
def test_update_and_retrieve_cache(kv_cache, batch_size, num_heads, seq_len, head_dim):
    """Test single update and retrieval of cache."""
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

    updated_keys, updated_values = kv_cache.update(key_states, value_states, layer_idx=0)

    assert updated_keys.shape == (batch_size, num_heads, seq_len, head_dim)
    assert updated_values.shape == (batch_size, num_heads, seq_len, head_dim)
    assert kv_cache.get_cache_length(0) == seq_len

@pytest.mark.parametrize(
    "seq_len1, seq_len2",
    [
        (3, 5),
        (1, 2),
    ]
)
def test_concatenated_update_shapes(kv_cache, seq_len1, seq_len2):
    """Test that consecutive updates correctly concatenate along sequence length."""
    batch_size, num_heads, head_dim = 2, 4, 8

    key_states1 = torch.randn(batch_size, num_heads, seq_len1, head_dim)
    value_states1 = torch.randn(batch_size, num_heads, seq_len1, head_dim)
    kv_cache.update(key_states1, value_states1, layer_idx=1)

    key_states2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
    value_states2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
    updated_keys, updated_values = kv_cache.update(key_states2, value_states2, layer_idx=1)

    total_seq_len = seq_len1 + seq_len2
    assert updated_keys.shape[-2] == total_seq_len
    assert updated_values.shape[-2] == total_seq_len
    assert kv_cache.get_cache_length(1) == total_seq_len

def test_invalid_layer_access(kv_cache):
    """Test accessing a non-existent layer raises IndexError."""
    with pytest.raises(IndexError):
        _ = kv_cache.get_cache_length(layer_idx=3)  # Only layers 0 and 1 exist

def test_cache_content_concatenation(kv_cache):
    """Test that cached content is correctly concatenated across updates."""
    batch_size, num_heads, head_dim = 1, 2, 4
    seq_len1 = 2
    seq_len2 = 3

    key_states1 = torch.randn(batch_size, num_heads, seq_len1, head_dim)
    value_states1 = torch.randn(batch_size, num_heads, seq_len1, head_dim)
    kv_cache.update(key_states1, value_states1, layer_idx=0)

    key_states2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
    value_states2 = torch.randn(batch_size, num_heads, seq_len2, head_dim)
    updated_keys, updated_values = kv_cache.update(key_states2, value_states2, layer_idx=0)

    # Check the first chunk matches key_states1
    assert torch.allclose(updated_keys[..., :seq_len1, :], key_states1, atol=1e-5)
    assert torch.allclose(updated_values[..., :seq_len1, :], value_states1, atol=1e-5)

    # Check the second chunk matches key_states2
    assert torch.allclose(updated_keys[..., seq_len1:, :], key_states2, atol=1e-5)
    assert torch.allclose(updated_values[..., seq_len1:, :], value_states2, atol=1e-5)
