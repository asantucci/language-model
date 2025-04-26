import torch
from model.moe.auxiliary_loss import AddAuxiliaryLoss

def test_auxiliary_loss_backward_pass():
    x = torch.randn(3, requires_grad=True)
    aux_loss = torch.tensor(0.0, requires_grad=True)

    out = AddAuxiliaryLoss.apply(x, aux_loss)
    out.sum().backward()

    assert x.grad is not None
    assert aux_loss.grad is not None
