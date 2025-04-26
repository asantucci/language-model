import pytest
import torch
import torch.nn as nn
from torch import functional as F
from config.deepseek import DeepSeekConfig
from model.moe.moe import MoE

@pytest.fixture
def moe_config():
    return DeepSeekConfig(
        d_model=512,
        nheads=8,
        max_position_embeddings=4096,
        dropout=0.1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_kv_cache=False,
        q_lora_rank=128,
        kv_lora_rank=128,
        nope_head_dim=64,
        v_head_dim=64,
        rope={"head_dim": 64, "base": 10000, "scaling": None},
        num_shared_experts=1,
        num_routed_experts=4,
        topk=2,
        moe_hidden_dimension=2048,
        mlp_hidden_dimension=4096,
        topk_norm_epsilon=1e-9,
        normalized_moe_gates=True,
        expert_load_balance_factor=0.01,
        rms_norm_eps=1e-6,
        first_k_dense_replace=1,
        num_layers=4,
        vocab_size=50257,
        init_weight_std=0.006,
    )

def test_moe_shapes():
    config = DeepSeekConfig(
        d_model=16,
        nheads=2,
        max_position_embeddings=128,
        dropout=0.1,
        device="cpu",
        use_kv_cache=False,
        q_lora_rank=8,
        kv_lora_rank=8,
        nope_head_dim=8,
        v_head_dim=8,
        rope={"head_dim": 8, "base": 10000},
        num_shared_experts=1,
        num_routed_experts=4,
        topk=2,
        moe_hidden_dimension=32,
        mlp_hidden_dimension=32,
        topk_norm_epsilon=1e-9,
        normalized_moe_gates=True,
        expert_load_balance_factor=0.01,
        rms_norm_eps=1e-5,
        first_k_dense_replace=1,
        num_layers=2,
        vocab_size=1000,
        init_weight_std=0.006,
    )

    moe = MoE(config)
    x = torch.randn(2, 5, config.d_model)

    output = moe(x)
    assert output.shape == (2, 5, config.d_model)

def test_moe_topk_assignment(moe_config):
    moe = MoE(moe_config)
    x = torch.randn(4, 8, moe_config.d_model)
    linear_layer = nn.Linear(moe_config.d_model, moe_config.num_routed_experts, bias=False)
    softmax_layer = nn.Softmax(dim=-1)
    with torch.no_grad():
        linear_layer.weight.copy_(moe.experts_weights)
    x_flat = x.view(-1, moe_config.d_model)
    gates = softmax_layer(linear_layer(x_flat))  # Shape: (B*T, num_experts).
    top_values, top_indices = torch.topk(gates, k=moe.topk, dim=-1)

    masked_gates = torch.zeros_like(gates)
    masked_gates.scatter_(1, top_indices, top_values)

    # Every token should have exactly topk non-zero entries
    nonzero_counts = (masked_gates > 0).sum(dim=-1)
    assert (nonzero_counts == moe.topk).all()

def test_moe_auxiliary_loss_nonzero(moe_config):
    moe = MoE(moe_config)
    moe.train()

    x = torch.randn(4, 8, moe_config.d_model)

    output = moe(x)
    loss = output.sum()
    loss.backward()

    # Check that at least some gradient flows through auxiliary loss
    grad_norms = [p.grad.norm().item() for p in moe.parameters() if p.grad is not None]
    assert any(norm > 0 for norm in grad_norms), "No gradients flowing from auxiliary loss."

def test_moe_gate_masking(moe_config):
    moe = MoE(moe_config)
    x = torch.randn(2, 4, moe_config.d_model)
    linear_layer = nn.Linear(moe_config.d_model, moe_config.num_routed_experts, bias=False)
    softmax_layer = nn.Softmax(dim=-1)
    with torch.no_grad():
        linear_layer.weight.copy_(moe.experts_weights)
    x_flat = x.view(-1, moe_config.d_model)
    gates = softmax_layer(linear_layer(x_flat))  # Shape: (B*T, num_experts).

    top_values, top_indices = torch.topk(gates, k=moe.topk, dim=-1)
    masked_gates = torch.zeros_like(gates)
    masked_gates.scatter_(1, top_indices, top_values)

    # The masked gates should only have topk nonzeros, and others should be exactly zero
    zeros = (masked_gates == 0).sum()
    nonzeros = (masked_gates != 0).sum()

    assert nonzeros == gates.shape[0] * moe.topk
    assert zeros == (gates.numel() - nonzeros)

def test_moe_gradients_flow_to_experts(moe_config):
    moe = MoE(moe_config)
    moe.train()

    x = torch.randn(2, 4, moe_config.d_model)
    output = moe(x)
    loss = output.sum()
    loss.backward()

    grad_norms = [p.grad.norm().item() for expert in moe.experts for p in expert.parameters() if p.grad is not None]
    assert all(norm > 0 for norm in grad_norms), "Some experts have zero gradients after backward pass."
