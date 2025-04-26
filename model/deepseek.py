import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config.deepseek import DeepSeekConfig
from model.moe.moe import MoE, FeedForward
from model.mhsa import MultiHeadSelfAttention, KVCache


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of:
      - Pre-normalized Multi-Head Self-Attention (MHSA)
      - Feedforward layer (either MoE or MLP)
      - Residual connections around each sub-layer

    Notes:
    - Pre-norm stabilization improves gradient flow in deep networks.
    - MoE is selectively activated after a certain number of dense blocks.
    """
    def __init__(self, config: DeepSeekConfig, block_idx: int):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.self_attn = MultiHeadSelfAttention(config, layer_idx=block_idx)

        if block_idx >= config.first_k_dense_replace:
            self.mlp = MoE(config)
        else:
            self.mlp = FeedForward(config.d_model, config.mlp_hidden_dimension)

        self.post_attention_layernorm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
    ):
        residual = x
        x, past_key_value = self.self_attn(self.input_layernorm(x), past_key_value)
        x = residual + x

        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = residual + x

        return x, past_key_value


class DeepSeekTransformer(nn.Module):
    """
    Backbone transformer stack for DeepSeek models.

    Responsibilities:
      - Token embeddings → transformer blocks → final layernorm
      - Causal mask and rotary embeddings handled inside attention layers

    Constraints:
      - Assumes input sequences fit within max_position_embeddings.
    """
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout, inplace=True)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, idx) for idx in range(config.num_layers)]
        )
        self.final_layernorm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.max_position_embeddings = config.max_position_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_value: Optional[KVCache] = None,
    ):
        """
        Args:
            input_ids (torch.Tensor): (batch_size, seq_len)
            past_key_value (KVCache, optional): for decoding

        Returns:
            output (torch.Tensor): (batch_size, seq_len, d_model)
            past_key_value (KVCache)
        """
        B, T = input_ids.shape
        assert T <= self.max_position_embeddings, (
            f"Input sequence length {T} exceeds model limit {self.max_position_embeddings}"
        )

        x = self.embed_tokens(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x, past_key_value = layer(x, past_key_value)

        x = self.final_layernorm(x)
        return x, past_key_value


class DeepSeekModelForCausalLM(nn.Module):
    """
    DeepSeek Transformer model for Causal Language Modeling (LM).

    Adds a linear LM head over transformer outputs.
    """
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.model = DeepSeekTransformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.topk = config.topk
        self.init_weight_std = config.init_weight_std
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights following DeepSeek initialization:
          - Normal(0, std) for all weights.
          - Zeros for biases (if applicable).
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_weight_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_value: Optional[KVCache] = None,
    ):
        """
        Args:
            input_ids (torch.Tensor): (batch_size, seq_len)
            targets (torch.Tensor, optional): labels for LM loss
            past_key_value (KVCache, optional): for efficient decoding

        Returns:
            logits (torch.Tensor): (batch_size, seq_len, vocab_size) during training, (batch_size, 1, vocab_size) during inference
            loss (torch.Tensor or None): cross-entropy loss if targets provided
            past_key_value (KVCache)
        """
        hidden_states, past_key_value = self.model(input_ids, past_key_value)

        if targets is not None:
            logits = self.lm_head(hidden_states)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(hidden_states[:, [-1], :])
            loss = None

        return logits, loss, past_key_value

    def get_total_parameters(self):
        """
        Returns:
            total_params (int): total number of parameters
            activated_params (int): parameters activated during top-k routing in MoE
        """
        total_params = 0
        activated_params = 0
        routed_moe_prefix = "mlp.experts"
        routed_moe_active = [f"{routed_moe_prefix}.{i}" for i in range(self.topk)]

        def is_active(name):
            return any(name.startswith(prefix) for prefix in routed_moe_active)

        for name, param in self.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                if routed_moe_prefix in name or is_active(name):
                    activated_params += param.numel()

        return total_params, activated_params    

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            input_ids (torch.Tensor): (batch_size, seq_len)
            max_length (int): number of tokens to generate
            temperature (float): sampling temperature (higher → more random)

        Returns:
            torch.Tensor: (batch_size, seq_len + max_length)
        """
        kv_cache = KVCache(self.config.num_layers)

        for _ in range(max_length):
            logits, _, kv_cache = self(input_ids, past_key_value=kv_cache)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
