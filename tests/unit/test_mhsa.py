import torch
import pytest
from config.deepseek import DeepSeekConfig
from model.mhsa import MultiHeadSelfAttention

@pytest.fixture
def mhsa_config():
    return DeepSeekConfig(
        d_model=512,
        nheads=8,
        max_position_embeddings=512,
        dropout=0.1,
        use_kv_cache=False,
        q_lora_rank=None,
        kv_lora_rank=64,
        nope_head_dim=32,
        v_head_dim=32,
        rope={
            "head_dim": 32,
            "base": 10000,
            "scaling": None,
        },
        num_shared_experts=2,
        num_routed_experts=4,
        topk=2,
        moe_hidden_dimension=512,
        mlp_hidden_dimension=2048,
        topk_norm_epsilon=1e-9,
        normalized_moe_gates=True,
        expert_load_balance_factor=0.01,
        rms_norm_eps=1e-6,
        first_k_dense_replace=1,
        num_layers=4,
        vocab_size=10000,
        init_weight_std=0.02,
        disable_moe=True,
        device='cpu'
    )

def test_mhsa_forward_no_kv_cache(mhsa_config):
    batch_size = 2
    seq_len = 16
    hidden_dim = mhsa_config.d_model

    mhsa = MultiHeadSelfAttention(mhsa_config, layer_idx=0)
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim)

    output, kv_cache = mhsa(input_tensor)

    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert kv_cache is None

def test_mhsa_forward_with_kv_cache(mhsa_config):
    batch_size = 2
    seq_len = 8
    hidden_dim = mhsa_config.d_model

    mhsa_config.use_kv_cache = True
    mhsa = MultiHeadSelfAttention(mhsa_config, layer_idx=0)
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim)

    output, kv_cache = mhsa(input_tensor, past_key_value=None)

    assert output.shape == (batch_size, seq_len, hidden_dim)
    # Check that KV cache has been initialized
    assert kv_cache is not None
    assert hasattr(kv_cache, "key_cache")
    assert hasattr(kv_cache, "value_cache")


def test_mhsa_forward_kv_cache_grows_across_steps(mhsa_config):
    batch_size = 2
    hidden_dim = mhsa_config.d_model

    mhsa_config.use_kv_cache = True
    mhsa = MultiHeadSelfAttention(mhsa_config, layer_idx=0)

    kv_cache = None
    total_seq_len = 0

    for step_seq_len in [2, 3, 5]:  # Simulate incremental forward passes
        x = torch.randn(batch_size, step_seq_len, hidden_dim)
        output, kv_cache = mhsa(x, past_key_value=kv_cache)

        total_seq_len += step_seq_len

        # After each forward, kv_cache for this layer should have grown accordingly
        assert kv_cache is not None
        cached_keys = kv_cache.key_cache[mhsa.layer_idx]
        cached_values = kv_cache.value_cache[mhsa.layer_idx]

        assert cached_keys is not None
        assert cached_keys.shape[-2] == total_seq_len
        assert cached_values.shape[-2] == total_seq_len

        # Check that shapes are otherwise consistent
        assert cached_keys.shape[0] == batch_size
        assert cached_keys.shape[1] == mhsa.nheads
        assert cached_keys.shape[-1] == mhsa.q_head_dim
        assert cached_values.shape[0] == batch_size
        assert cached_values.shape[1] == mhsa.nheads
        assert cached_values.shape[-1] == mhsa.v_head_dim