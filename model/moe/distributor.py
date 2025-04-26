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

        Given that each token may be routed to multiple experts (top-k routing),
        this function:
          1. Duplicates tokens if needed.
          2. Groups all tokens belonging to the same expert together.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size * sequence_length, hidden_dim).

        Returns:
            List[torch.Tensor]: A list of tensors, one per expert.
                Each tensor contains the expert's assigned tokens.
        Example:
            Suppose input x has 3 tokens:
                Token 0
                Token 1
                Token 2

            And token routing is:
                Token 0 -> Expert 1
                Token 1 -> Expert 0 and Expert 1
                Token 2 -> Expert 0

            Then expanded_x will look like:

                Index 0: Token 1  (for Expert 0)
                Index 1: Token 2  (for Expert 0)
                Index 2: Token 0  (for Expert 1)
                Index 3: Token 1  (for Expert 1)

            Grouped into:
                Expert 0's input: [Token 1, Token 2]
                Expert 1's input: [Token 0, Token 1]
        """
        # After indexing, because each token may appear multiple times if it's being routed to
        # multiple experts (topk > 1), the shape is `B*T*topk, hidden_dim`.
        expanded_x = x[self._batch_indices]
        return expanded_x.split(self._groups)

    def combine(self, expert_outputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Combine the outputs from all experts back into the original token order.

        Steps:
            1. Concatenate all expert outputs into a single long tensor.
            2. Weight each expert output according to the original softmax gates.
            3. Scatter-add the weighted outputs back to their original token positions.

        Args:
            expert_outputs (list[torch.Tensor]): List of expert outputs, each of varying batch size.

        Returns:
            torch.Tensor: Combined output tensor of shape (batch_size * sequence_length, hidden_dim).
        
        Example:
            Continuing the example from `prepare_inputs_for_experts`:

                Expert 0 output: [y1, y2]  # For Token 1, Token 2
                Expert 1 output: [y0, y1]  # For Token 0, Token 1

            After concatenation:

                [y1 (Token 1, Expert 0),
                 y2 (Token 2, Expert 0),
                 y0 (Token 0, Expert 1),
                 y1 (Token 1, Expert 1)]

            Weighted by softmax scores and then scattered back:

                Final output:
                    Token 0 = y0 * weight0
                    Token 1 = y1_from_expert0 * weight1 + y1_from_expert1 * weight2
                    Token 2 = y2 * weight3

        Note:
            - Token 1 receives contributions from multiple experts (index_add handles summing).
            - Batch indices are reused because each token may have multiple expert contributions.
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
