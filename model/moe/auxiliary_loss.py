import torch

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    Injects auxiliary loss (e.g., expert balancing loss) into the computation graph without affecting forward outputs.

    Custom autograd function to attach auxiliary losses to the computation graph.

    In Mixture-of-Experts (MoE) and other models, some losses (like load balancing penalties)
    should affect optimization but should not modify the forward output of the model. The functionality we desire
    is therefore: (a) during forward pass we return the original tensor unchanged, and (b) during backward pass,
    we add the gradient of the auxiliary loss to the total loss automatically.

    Args:
        x (Tensor): The primary output tensor.
        loss (Tensor): A scalar auxiliary loss (must have shape [1]).

    Returns:
        Tensor: The original `x` tensor, untouched in forward pass, with loss attached in backward.
    """

    @staticmethod
    def forward(ctx, x, aux_loss):
        assert aux_loss.numel() == 1
        ctx.dtype = aux_loss.dtype
        ctx.required_aux_loss = aux_loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_aux = None
        if ctx.required_aux_loss:
            grad_aux = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_aux
