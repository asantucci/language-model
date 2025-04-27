import torch
from model.moe.distributor import Distributor

def test_distributor_shapes_and_split():
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_experts = 3
    topk = 2

    x = torch.randn(batch_size * seq_len, d_model)
    gates = torch.zeros(batch_size * seq_len, num_experts)
    # Simulate a routing: every token routes to two experts
    for i in range(batch_size * seq_len):
        gates[i, i % num_experts] = 0.6
        gates[i, (i + 1) % num_experts] = 0.4

    distributor = Distributor(gates, topk, 'cpu')
    expert_inputs = distributor.prepare_inputs_for_experts(x)

    # Check that we split into the right number of experts
    assert len(expert_inputs) == num_experts
    # The sum of expert tokens should equal total tokens routed
    total_tokens = sum(inp.shape[0] for inp in expert_inputs)
    assert total_tokens == (batch_size * seq_len * topk)

def test_distributor_combine():
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_experts = 2
    topk = 1

    x = torch.randn(batch_size * seq_len, d_model)
    gates = torch.zeros(batch_size * seq_len, num_experts)
    for i in range(batch_size * seq_len):
        gates[i, i % num_experts] = 1.0

    distributor = Distributor(gates, topk, 'cpu')
    expert_inputs = distributor.prepare_inputs_for_experts(x)
    outputs = [inp + 1 for inp in expert_inputs]

    combined = distributor.combine(outputs)
    assert combined.shape == (batch_size * seq_len, d_model)
