import torch
import pytest
from model.deepseek import DeepSeekConfig, DeepSeekModelForCausalLM
from model.mhsa import KVCache


@pytest.fixture
def dummy_config():
    return DeepSeekConfig(
        d_model=32,
        nheads=4,
        max_position_embeddings=128,
        dropout=0.1,
        device="cpu",
        use_kv_cache=True,
        q_lora_rank=None,
        kv_lora_rank=8,
        nope_head_dim=8,
        v_head_dim=8,
        rope={"head_dim": 8, "base": 10000, "scaling": None},
        num_shared_experts=1,
        num_routed_experts=4,
        topk=2,
        moe_hidden_dimension=64,
        mlp_hidden_dimension=64,
        topk_norm_epsilon=1e-9,
        normalized_moe_gates=True,
        expert_load_balance_factor=0.01,
        rms_norm_eps=1e-6,
        first_k_dense_replace=0,
        num_layers=2,
        vocab_size=100,
        init_weight_std=0.02,
    )


@pytest.fixture
def model(dummy_config):
    return DeepSeekModelForCausalLM(dummy_config)


def test_forward_shape(model):
    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    logits, loss, _ = model(input_ids, targets=input_ids)

    assert logits.shape == (batch_size, seq_len, model.config.vocab_size)
    assert loss is not None


def test_generation_shapes(model):
    batch_size = 2
    seq_len = 5
    max_gen_len = 10

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    output_ids = model.generate(input_ids, max_length=max_gen_len)

    assert output_ids.shape == (batch_size, seq_len + max_gen_len)


def test_forward_with_kv_cache(model):
    batch_size = 1
    seq_len = 8

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    kv_cache = KVCache(num_layers=model.config.num_layers)

    logits, _, updated_cache = model(input_ids, past_key_value=kv_cache)

    assert logits.shape == (batch_size, 1, model.config.vocab_size)
    for cache in updated_cache.key_cache:
        assert cache is not None


def test_total_parameters(model):
    total_params, activated_params = model.get_total_parameters()
    assert total_params > 0
    assert activated_params > 0
    assert activated_params <= total_params


def test_autoregressive_stepwise_growth(model):
    batch_size = 1
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, 1))
    kv_cache = KVCache(num_layers=model.config.num_layers)

    total_steps = 5
    generated_tokens = []

    for _ in range(total_steps):
        logits, _, kv_cache = model(input_ids, past_key_value=kv_cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        input_ids = next_token

    assert len(generated_tokens) == total_steps
