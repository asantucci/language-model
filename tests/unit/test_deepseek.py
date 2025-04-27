import torch
import pytest
from model.deepseek import DeepSeekConfig, DeepSeekModelForCausalLM
from model.mhsa import KVCache
from tests.conftest import dummy_args


@pytest.fixture
def model(dummy_args):
    return DeepSeekModelForCausalLM(dummy_args)


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
    assert activated_params >= 0
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
