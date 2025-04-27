import json

from dataclasses import asdict, dataclass

@dataclass
class DeepSeekConfig:
    # Model architecture parameters
    d_model: int
    nheads: int
    max_position_embeddings: int
    dropout: float
    use_kv_cache: bool
    q_lora_rank: int
    kv_lora_rank: int
    nope_head_dim: int
    v_head_dim: int
    num_shared_experts: int
    num_routed_experts: int
    moe_hidden_dimension: int
    mlp_hidden_dimension: int
    topk: int
    topk_norm_epsilon: float
    rms_norm_eps: float
    normalized_moe_gates: bool
    expert_load_balance_factor: float
    num_layers: int
    vocab_size: int
    init_weight_std: float
    first_k_dense_replace: int

    # RoPE settings
    rope: dict

    # For testing, we disable MoE since we don't pad the batch to the next multiple of 8 tokens, e.g.,
    # and we don't ensure top-k gating is normalized, and we don't yet consider token dropout during
    # backscatter.
    disable_moe: bool

    device: str = "cuda"

    @staticmethod
    def from_json(path: str) -> "DeepSeekConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return DeepSeekConfig(**data)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
