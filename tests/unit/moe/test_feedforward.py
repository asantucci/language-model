import torch
from model.moe.feedforward import FeedForward

def test_feedforward_shapes():
    d_model = 16
    hidden_dim = 32
    ff = FeedForward(d_model, hidden_dim)
    x = torch.randn(4, d_model)

    out = ff(x)
    assert out.shape == (4, d_model)
