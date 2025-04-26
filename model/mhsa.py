import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from typing import Optional

from config.deepseek import DeepSeekConfig
from model.lora import LoRALinear
from model.kv_cache import KVCache
from model.rope import RoPE

def get_causal_mask(seq_len: int, device: str) -> torch.Tensor:
    """
    Returns a lower triangular causal mask of shape (1, 1, seq_len, seq_len).
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.bool().unsqueeze(0).unsqueeze(1)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention layer with support for LoRA, RoPE, KV caching.
    """

    def __init__(self, config: DeepSeekConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Projections for Q, K, V
        self.q_head_dim = config.rope['head_dim'] + config.nope_head_dim
        self.kv_head_dim = config.nope_head_dim + config.v_head_dim

        # Query projections (LoRA if enabled)
        self.q_lora_rank = config.q_lora_rank
        self.q_proj = LoRALinear(config.d_model, config.nheads * self.q_head_dim, lora_rank=config.q_lora_rank)

        # Key/Value projections
        self.kv_a_proj_with_mqa = nn.Linear(config.d_model, config.kv_lora_rank + config.rope['head_dim'], bias=False)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.nheads * self.kv_head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(config.nheads * config.v_head_dim, config.d_model, bias=False)

        # Dropouts
        self.attention_dropout_rate = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)

        # Model shape settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nheads = config.nheads
        self.rope_head_dim = config.rope['head_dim']
        self.nope_head_dim = config.nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.max_position_embeddings = config.max_position_embeddings

        # RoPE object
        self._init_rope()

    def _init_rope(self):
        rope_config = self.config.rope
        scaling = rope_config.get('scaling')

        if scaling is None:
            self.rope = RoPE(
                embed_dim=self.rope_head_dim,
                base=rope_config['base'],
            )
        else:
            raise ValueError(f"Unsupported RoPE scaling type: {scaling}")

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
    ):
        """
        Args:
            x: input tensor of shape (B, T, d_model)
            past_key_value: optional KVCache for decoding
        Returns:
            output: tensor of shape (B, T, d_model)
            updated past_key_value
        """
        B, q_len, _ = x.shape

        # Project Queries
        if self.q_lora_rank is not None:
            q = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(q)
        else:
            q = self.q_proj(x)
        q = q.view(B, q_len, self.nheads, self.q_head_dim).transpose(1, 2)

        q_nope, q_rope = torch.split(q, [self.nope_head_dim, self.rope_head_dim], dim=-1)

        # Project Keys and Values
        kv_compressed = self.kv_a_proj_with_mqa(x)
        kv_compressed, k_rope = kv_compressed.split([self.config.kv_lora_rank, self.rope_head_dim], dim=-1)
        k_rope = k_rope.view(B, 1, q_len, self.rope_head_dim)

        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed)).view(B, -1, self.nheads, self.kv_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.nope_head_dim, self.v_head_dim], dim=-1)

        past_seq_len = past_key_value.get_cache_length(self.layer_idx) if past_key_value is not None else 0
        kv_seq_len = past_seq_len + q_len

        # Make sure the RoPE cache is large enough
        self.rope._build_cache(kv_seq_len)
        q_rope = self.rope(q_rope, offset=past_seq_len)
        k_rope = self.rope(k_rope, offset=past_seq_len)

        # Merge Q
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope.expand_as(k_nope)], dim=-1)

        # Update KV cache if possible, otherwise initialize it from scratch.
        if past_key_value is None and self.config.use_kv_cache:
            past_key_value = KVCache(num_layers=self.config.num_layers)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )
        
        # Build causal mask if needed
        attn_mask = None
        if q_len == kv_seq_len:
            attn_mask = get_causal_mask(seq_len=kv_seq_len, device=x.device)

        # Attention
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout_rate if self.training else 0.0,
        )

        # Final projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, q_len, -1)
        output = self.residual_dropout(self.o_proj(attn_output))

        return output, past_key_value
