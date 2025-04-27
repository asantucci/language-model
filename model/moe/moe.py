import torch
import torch.nn as nn
import torch.nn.functional as F

from model.moe.distributor import Distributor
from model.moe.feedforward import FeedForward
from model.moe.auxiliary_loss import AddAuxiliaryLoss
from config.deepseek import DeepSeekConfig

class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) layer with shared and routed experts.

    In a Mixture-of-Experts system, parallelism depends on how tokens are *grouped* and *routed*.
    Naive implementations consist of passing each token indvidually through to its assigned expert.
    Efficient implementations will batch tokens per expert using a Distributor class.
      - All tokens assigned to expert 0 are gathered into one tensor, processed in a single matrix multiply.
      - All tokens assigned to expert 1 are gathered and processed in a separate batch, etc.
    This effectively allows a singly matmul within an expert across all tokens assigned to that expert.
    Since GPU's are optimized for matmuls on contiguous blocks of data, this increases efficiency in practice.
    """
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_shared_experts = config.num_shared_experts
        self.moe_hidden_dimension = config.moe_hidden_dimension
        self.topk = config.topk
        self.num_routed_experts = config.num_routed_experts

        self.experts_weights = nn.Parameter(
            torch.randn(self.num_routed_experts, config.d_model) * config.init_weight_std
        )

        self.experts = nn.ModuleList([
            FeedForward(config.d_model, self.moe_hidden_dimension)
            for _ in range(self.num_routed_experts)
        ])

        self.shared_experts = FeedForward(
            config.d_model, self.moe_hidden_dimension * self.num_shared_experts
        )

        self.topk_norm_epsilon = config.topk_norm_epsilon
        self.normalized_moe_gates = config.normalized_moe_gates
        self.expert_load_balance_factor = config.expert_load_balance_factor
        self.device = config.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_optimized(x)

    def _forward_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optimized forward pass for Mixture-of-Experts (MoE).

        This method batches tokens per expert to maximize efficiency. It avoids looping over tokens
        by:
          - Grouping all tokens routed to the same expert
          - Processing each expert's batch independently
          - Recombining expert outputs using the Distributor class
          - Adding shared expert output (processed for all tokens)

        It also injects an auxiliary load balancing loss during training.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model).
        """
        B, T, d_model = x.size()
        x_flat = x.view(B * T, d_model)

        gates = F.linear(x_flat, self.experts_weights)
        gates = F.softmax(gates, dim=-1)

        top_values, top_indices = torch.topk(gates, k=self.topk, dim=-1, sorted=False)
        masked_gates = torch.zeros_like(gates)
        masked_gates = torch.scatter(masked_gates, 1, top_indices, top_values)

        if self.normalized_moe_gates:
            masked_gates = masked_gates / (masked_gates.sum(dim=-1, keepdim=True) + self.topk_norm_epsilon)

        distributor = Distributor(masked_gates, self.topk, x.device)

        expert_inputs = distributor.prepare_inputs_for_experts(x_flat)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_routed_experts)]

        routed_combined = distributor.combine(expert_outputs).view(B, T, -1)

        if self.training:
            masked_gates = masked_gates.view(B, T, -1)
            gates = gates.view(B, T, -1)

            load = (masked_gates > 0).sum(dim=1)
            expert_prob_sum = gates.sum(dim=1)

            load_balance_loss = self.expert_load_balance_factor * (
                (self.num_routed_experts / (self.topk * T) * load) * (1.0 / T * expert_prob_sum)
            ).sum(dim=1).mean()

            routed_combined = AddAuxiliaryLoss.apply(routed_combined, load_balance_loss)

        shared_output = self.shared_experts(x_flat).view(B, T, -1)
        return routed_combined + shared_output
