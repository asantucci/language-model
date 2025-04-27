"""Caching class to support fast(er) autoregressive decoding in the context of attention."""
import torch

class KVCache:
    """
    Maintains a key-value memory for each Transformer layer during inference or generation.
    
    Designed to support fast autoregressive decoding by caching previously computed
    key and value tensors across multiple forward passes.

    Attributes:
        key_cache (List[Optional[torch.Tensor]]): List of key tensors, one per layer.
        value_cache (List[Optional[torch.Tensor]]): List of value tensors, one per layer.
    """

    def __init__(self, num_layers: int):
        """
        Args:
            num_layers (int): Number of Transformer layers to maintain cache for.
        """
        self.key_cache = [None] * num_layers
        self.value_cache = [None] * num_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenates new states with existing cached states along the sequence length dimension.

        Args:
            key_states (torch.Tensor): New key tensor to add (shape [B, H, T, D]).
            value_states (torch.Tensor): New value tensor to add (shape [B, H, T, D]).
            layer_idx (int): Layer index to update.

        Returns:
            Tuple of updated key and value tensors for the layer.
        """
        past_keys = self.key_cache[layer_idx]
        past_values = self.value_cache[layer_idx]

        if past_keys is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat((past_keys, key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((past_values, value_states), dim=-2)

        # Force cache to be on same device as incoming key_states
        self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_cache_length(self, layer_idx: int) -> int:
        """
        Returns the current sequence length of cached keys/values for a given layer.

        Args:
            layer_idx (int): Layer index to query.

        Returns:
            int: Cached sequence length. Returns 0 if cache is empty.
        """
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]
