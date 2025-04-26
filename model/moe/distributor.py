import torch

class Distributor:
    """
    Groups and redistributes tokens across experts based on routing scores.

    Given a sparse gate matrix (only top-k nonzeros), Distributor:
      - reorders input tokens so each expert sees its routed tokens as a contiguous block
      - gathers expert outputs
      - combines expert outputs back to the original token order using weighted sums
    """
    def __init__(self, gates: torch.Tensor, topk: int):
        """
        Initialize the Distributor.

        Args:
            gates (torch.Tensor): A tensor of shape (batch_size * sequence_length, num_experts)
                containing sparse softmax probabilities. Only top-k entries per token are nonzero.
            topk (int): The number of experts each token is routed to.
        """
        self.topk = topk
        batch_and_expert_indices = torch.nonzero(gates)
        sorted_expert_indices, index_sorted = batch_and_expert_indices.sort(dim=0, stable=True)

        old_expert_indices = index_sorted[:, 1]
        self._batch_indices = batch_and_expert_indices[:, 0][old_expert_indices]
        self._groups = (gates > 0).sum(dim=0).tolist()
        self._weights = gates.t().reshape(-1)[gates.t().reshape(-1) > 0].view(-1, 1)

    def prepare_inputs_for_experts(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Reorder and split the input tokens so each expert receives a contiguous tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size * sequence_length, hidden_dim).

        Returns:
            List[torch.Tensor]: A list of tensors, one per expert.
                Each tensor contains the expert's assigned tokens.
        """
        expanded_x = x[self._batch_indices]
        return expanded_x.split(self._groups)

    def combine(self, expert_outputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Combine the outputs from all experts back into the original token order.

        Args:
            expert_outputs (list[torch.Tensor]): List of expert outputs, each of varying batch size.

        Returns:
            torch.Tensor: Combined output tensor of shape (batch_size * sequence_length, hidden_dim).
        """
        combined = torch.cat(expert_outputs, dim=0)
        combined = combined * self._weights
        output = torch.zeros(
            combined.shape[0] // self.topk, 
            combined.shape[1],
            dtype=combined.dtype,
            device=combined.device
        )
        output.index_add_(0, self._batch_indices, combined)
        return output
