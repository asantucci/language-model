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

    A note on NOPE (Non-Rotary Positional Embedding).

    In the DeepSeek attention architecture, query and key projections are split into two components:
    - A "ROPE" portion that receives rotary positional encoding (relative position-aware).
    - A "NOPE" portion that receives no positional encoding (position-agnostic).

    The NOPE subspace allows part of the attention mechanism to focus purely on content-based interactions,
    independent of token positions. This provides the model with the flexibility to:
    - Learn relationships where relative or absolute positions are unimportant.
    - Improve generalization across varying sequence lengths.
    - Support retrieval-based and factual tasks that depend more on content matching than position.

    Implementation Details:
    - After linear projections, the head dimension is split into `nope_head_dim` and `rope_head_dim`.
    - Rotary position embeddings are applied only to the ROPE portion.
    - The NOPE portion remains unaltered throughout.
    - The two components are concatenated before computing scaled dot-product attention.

    NOPE complements ROPE by offering a mix of positional and non-positional inductive biases
    within the same multi-head attention layer. Increasing `nope_head_dim` in the configuration
    # may improve generalization on tasks where position is less important (e.g. retrieval, factual QA,
    and document search). In contrast, increasing `rope_head_dim` would favor tasks which have a strong
    sequential structure (e.g. summarization, or step-by-step reasoning).
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
        Multi-Head Self-Attention (MHSA) forward pass with support for rotary embeddings, LoRA compression, 
        and optional key-value (KV) caching for fast autoregressive decoding.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, T, d_model), where
                - B = batch size
                - T = input sequence length
                - d_model = hidden dimension of model

            past_key_value (Optional[KVCache]):
                Optional pre-computed key and value tensors from previous steps.
                If provided, enables fast autoregressive inference by avoiding recomputing past keys/values.
                If None, a fresh cache is created if config.use_kv_cache is True.

        Returns:
            output (torch.Tensor):
                Output tensor of shape (B, T, d_model) after self-attention and final projection.

            updated_past_key_value (KVCache or None):
                Updated KVCache containing accumulated keys and values if caching is used.
        """
        B, q_len, _ = x.shape

        q = self.q_proj(x)  # Applies LoRA if applicable.
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
        k_rope = k_rope.expand(-1, self.nheads, -1, -1)  # Expand heads dimension!
        key_states = torch.cat([k_nope, k_rope], dim=-1)

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

        # Attention operation. Note that this can be implemented from scratch as well, but that
        # torch.functional is going to be more performant as it fuses operations into one GPU kernel launch.
        # In contrast, the `scaled_dot_product_attention` written from scratch requires a separate GPU kernel launch
        # for each operation: einsum, matmul, scaling by \sqrt(d), adding mask, softmax, and matmul.
        # 
        # def scaled_dot_product_attention(self, Q, K, V, head_dim, mask=None):
        #     K_T = np.einsum('bhkd -> bhdk', K) / np.sqrt(head_dim)
        #     logits = np.matmul(Q, K_T)  # bhsd x bhdk --> bhsk (Treats first two dim as batch[es], matmul's the last two dims)
        #     if mask is not None:
        #         logits += mask
        #     attention_weights = self.softmax(logits)
        #     return np.matmul(attention_weights, V)
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
