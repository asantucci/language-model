import torch
import torch.nn as nn
from torch.nn import RMSNorm

class LoRALinear(nn.Module):
    """
    In large neural networks, especially Transformers, the weight matrices in attention and MLP layers are of size:
        - W ∈ R^{d × d}, where d = hidden dimension (e.g., 4096, 5120).

    When training a model:
        - Updating W directly requires O(d^2) parameters and O(d^2) computational cost per forward and backward pass.
        - Storage of optimizer states (like Adam's momentum and variance) also scales as O(d^2).

    LoRA proposes replacing the full weight update with a low-rank approximation:
        - Instead of directly updating W, we learn two smaller matrices:
            - A ∈ R^{r × d}
            - B ∈ R^{d × r}
        - The adapted weight is then: W' = W + BA
        - where r << d (r is the rank, e.g., 4, 8, 16).

    Parameter and compute savings:
        - Full update: O(d^2) parameters
        - LoRA update: O(2dr) parameters
        - Thus, parameter reduction ratio ≈ O(r/d).

    Memory savings:
        - Gradients, optimizer states, and activations for backpropagation are stored for A and B instead of W.

    Compute savings:
        - Multiplying by two thin matrices (B and A) is much cheaper than multiplying by a full square matrix W.
        - The new compute cost per update is O(dr) instead of O(d^2).

    Intuitive result:
        - LoRA allows effective fine-tuning with <1% of the original parameter count and similar expressiveness.
        - The original pretrained W is frozen, and only the lightweight adapters (B, A) are updated.
    """
    def __init__(self, input_dim, output_dim, lora_rank=None):
        super().__init__()
        self.lora_rank = lora_rank

        if lora_rank is not None:
            self.lora_a = nn.Linear(input_dim, lora_rank, bias=False)
            self.lora_norm = RMSNorm(lora_rank)
            self.lora_b = nn.Linear(lora_rank, output_dim, bias=False)
        else:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        if self.lora_rank is not None:
            x = self.lora_a(x)
            x = self.lora_norm(x)
            x = self.lora_b(x)
            return x
        else:
            return self.linear(x)
