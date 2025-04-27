import pytest
from config.deepseek import DeepSeekConfig

def make_dummy_args(**overrides):
    """Factory for dummy training args with sensible defaults, allowing overrides."""
    defaults = {
        "d_model": 128,
        "nheads": 2,
        "max_position_embeddings": 512,
        "dropout": 0.1,
        "use_kv_cache": True,
        "q_lora_rank": 64,
        "kv_lora_rank": 32,
        "nope_head_dim": 16,
        "rope": {
            "head_dim": 64,
            "base": 10000
        },
        "v_head_dim": 8,
        "num_shared_experts": 2,
        "num_routed_experts": 8,
        "moe_hidden_dimension": 64,
        "mlp_hidden_dimension": 64,
        "topk": 3,
        "topk_norm_epsilon": 1e-9,
        "rms_norm_eps": 1e-6,
        "normalized_moe_gates": True,
        "expert_load_balance_factor": 0.01,
        "num_layers": 2,
        "vocab_size": 1024,
        "init_weight_std": 0.006,
        "first_k_dense_replace": 1,
        "disable_moe": True,
        "device": "cuda"
    }
    defaults.update(overrides)
    return DeepSeekConfig(**defaults)

@pytest.fixture
def dummy_args():
    """Standard dummy args fixture."""
    return make_dummy_args()

@pytest.fixture
def tiny_batch_args():
    """Fixture for very small batches (for fast integration tests)."""
    return make_dummy_args(batch_size=2, seq_len=8)

@pytest.fixture
def fast_training_args():
    """Fixture for fast 1-2 step training runs."""
    return make_dummy_args(max_train_steps=2, batch_size=2, seq_len=8)
