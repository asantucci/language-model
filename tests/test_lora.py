import torch
import torch.nn as nn
import pytest
from model.lora import LoRALinear

@pytest.mark.parametrize("lora_rank", [None, 8])
def test_lora_linear_shapes_and_forward(lora_rank):
    batch_size = 4
    input_dim = 64
    output_dim = 128

    model = LoRALinear(input_dim, output_dim, lora_rank=lora_rank)
    x = torch.randn(batch_size, input_dim)

    y = model(x)

    assert y.shape == (batch_size, output_dim), "Output shape mismatch."

    if lora_rank is None:
        # Should have a vanilla Linear layer
        assert hasattr(model, "linear"), "Expected vanilla Linear layer when lora_rank=None."
        assert not hasattr(model, "lora_a"), "Unexpected LoRA layers when lora_rank=None."
    else:
        # Should have LoRA layers
        assert hasattr(model, "lora_a"), "Missing lora_a projection when lora_rank > 0."
        assert hasattr(model, "lora_norm"), "Missing lora_norm when lora_rank > 0."
        assert hasattr(model, "lora_b"), "Missing lora_b projection when lora_rank > 0."
        assert not hasattr(model, "linear"), "Unexpected vanilla Linear layer when lora_rank > 0."

@pytest.mark.parametrize("lora_rank", [None, 8])
def test_lora_linear_gradients(lora_rank):
    batch_size = 2
    input_dim = 32
    output_dim = 64

    model = LoRALinear(input_dim, output_dim, lora_rank=lora_rank)
    x = torch.randn(batch_size, input_dim, requires_grad=True)

    y = model(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Input should have gradients flowing back."

@pytest.mark.parametrize("lora_rank", [None, 4, 16])
def test_lora_linear_parameter_count(lora_rank):
    input_dim = 32
    output_dim = 64
    model = LoRALinear(input_dim, output_dim, lora_rank=lora_rank)

    if lora_rank is None:
        param_count = sum(p.numel() for p in model.parameters())
        expected = input_dim * output_dim  # standard Linear layer without bias
        assert param_count == expected, f"Expected {expected} params, got {param_count}"
    else:
        param_count = sum(p.numel() for p in model.parameters())
        expected = (input_dim * lora_rank) + (lora_rank) + (lora_rank * output_dim)
        # input to rank -> RMSNorm params -> rank to output
        assert abs(param_count - expected) < 10, f"Expected ~{expected} params, got {param_count}"
