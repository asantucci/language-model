from dataclasses import dataclass, asdict
import json
from typing import Dict, Optional

"""
DeepSeek V2 Model Configuration

Reference:
    - d_model: 5120             (hidden size)
    - nheads: 128               (number of attention heads)
    - block_size: 4096 → 128k    (context window)
    - q_lora_rank: 1536         (query LoRA rank, d_c_prime)
    - kv_lora_rank: 512         (key/value LoRA rank, d_c)
    - rope_head_dim: 64         (RoPE embedding dim per head, d_h_R)
    - nope_head_dim: 128        (non-RoPE head dimension, d_h)
"""

@dataclass
class DeepSeekConfig:
    # Core model dimensions
    d_model: int                          # Hidden dimension
    nheads: int                           # Number of attention heads
    max_position_embeddings: int          # Maximum sequence length
    dropout: float                         # Dropout rate
    device: str                            # 'cuda' or 'cpu'
    use_kv_cache: bool                     # Whether to enable KV caching

    # Attention (MHSA) parameters
    q_lora_rank: Optional[int]             # LoRA rank for query projection (optional)
    kv_lora_rank: int                      # LoRA rank for key/value projection
    nope_head_dim: int                     # Dimension for non-RoPE head features
    v_head_dim: int                        # Value head dimension
    rope: Dict                             # Rotary positional embedding settings

    # Mixture of Experts (MoE) parameters
    num_shared_experts: int
    num_routed_experts: int
    topk: int                              # Top-k experts selected
    moe_hidden_dimension: int              # Expert hidden dimension
    mlp_hidden_dimension: int              # Feedforward hidden dimension
    topk_norm_epsilon: float               # Epsilon for gate normalization
    normalized_moe_gates: bool             # Whether to normalize MoE gates
    expert_load_balance_factor: float      # Load balancing coefficient (alpha1)

    # Layer normalization / Initialization
    rms_norm_eps: float                    # RMSNorm epsilon
    first_k_dense_replace: int             # How many first dense layers are replaced
    init_weight_std: float                 # Std dev of weight initialization

    # Architecture structure
    num_layers: int                        # Total number of transformer layers
    vocab_size: int                        # Vocabulary size

    @staticmethod
    def from_json(path: str) -> "DeepSeekConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return DeepSeekConfig(**data)

    def to_json(self, path: str):
        """Save configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)