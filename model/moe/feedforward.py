import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Simple feedforward expert used inside MoE.

    DeepSeek-specific variant: SiLU(gate_proj(x)) * up_proj(x) -> down_proj
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
