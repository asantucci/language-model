import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE) module.

    This module applies position-dependent rotations to the input embeddings,
    enabling models to encode relative positional information in self-attention.

    Attributes:
        embed_dim (int): Total embedding dimension (must be divisible by 2).
        base (float): Base used to compute inverse frequencies.
        half (int): Half of the embedding dimension, used for splitting even/odd parts.
        cos_cache (torch.Tensor or None): Cached cosine values for given sequence lengths.
        sin_cache (torch.Tensor or None): Cached sine values for given sequence lengths.
    """
    def __init__(self, embed_dim: int, base: int = 10_000):
        """
        Args:
            embed_dim (int): Dimensionality of input features (must be even).
            base (float, optional): Base value for frequency scaling. Default is 10,000.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.base = base
        self.half = embed_dim // 2

        self.register_buffer('cos_cache', None, persistent=False)
        self.register_buffer('sin_cache', None, persistent=False)

    def _build_cache(self, seq_len: int):
        """
        Build and cache sin/cos tables for given sequence length.

        Args:
            seq_len (int): Sequence length to precompute sin/cos tables for.
        """
        if self.cos_cache is not None and seq_len <= self.cos_cache.shape[-2]:
            return  # Already cached for this or larger seq_len.

        positions = torch.arange(seq_len)
        frequencies = 1.0 / (self.base ** (torch.arange(0, self.half).float() / self.half))
        angles = torch.outer(positions, frequencies)  # Shape: (seq_len, embed_dim / 2)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Duplicate cos/sin across even and odd dimensions
        self.cos_cache = torch.cat([cos, cos], dim=-1)[None, None, :, :]  # (1, 1, seq_len, embed_dim)
        self.sin_cache = torch.cat([sin, sin], dim=-1)[None, None, :, :]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the dimensions (swap and negate).

        Args:
            x (torch.Tensor): Input tensor of shape (..., embed_dim).

        Returns:
            torch.Tensor: Rotated tensor of same shape as input.
        """
        x1, x2 = x[..., :self.half], x[..., self.half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary positional embeddings to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, heads, seq_len, embed_dim).
            offset: optional int for pre-existing cache length.

        Returns:
            torch.Tensor: Tensor with rotary embeddings applied.
        """
        batch, heads, seq_len, emb_dim = x.shape
        assert emb_dim == self.embed_dim, f"Expected last dim {self.embed_dim}, got {emb_dim}"

        self._build_cache(seq_len + offset)

        cos = self.cos_cache[..., offset:offset+seq_len, :]  # (1, 1, seq_len, embed_dim)
        sin = self.sin_cache[..., offset:offset+seq_len, :]

        return x * cos + self._rotate_half(x) * sin