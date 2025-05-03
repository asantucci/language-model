import itertools
from pathlib import Path
import random
import yaml

# Base directory for saving generated configs
CONFIG_DIR = Path("experiments/configs")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Define sweep search space
D_MODELS = [128, 256, 512, 1024, 2048]
LAYERS = [2, 4, 6, 8, 12]
HEADS = [1, 2, 4, 8, 16]
MOE_MLP_RATIO = [1.0, 0.5]  # moe_hidden_dimension = ratio * mlp_hidden_dimension
DROPOUTS = [0.05, 0.1, 0.2]
NOPE_DIM_OPTIONS = lambda rope_dim: list(range(0, rope_dim, rope_dim // 2 or 1))
RATIO_OF_CONFIGS_TO_KEEP = 0.001  # There are approximately 60k configs based on above. We sample ~60.

# Helper function to determine valid kv_lora_rank / rope_head_dim splits
def split_kv_capacity(kv_capacity):
    return [(r, kv_capacity - r) for r in range(0, kv_capacity + 1, 8)]  # 8-aligned splits

def generate_config(d_model, nheads, num_layers, moe_ratio, kv_lora_rank, rope_head_dim, nope_dim, dropout):
    kv_capacity = d_model // nheads
    moe_routers_exponent = random.randint(0, 5)
    config = {
        "kv_cache": {
            "use_kv_cache": False,
            "q_lora_rank": None,
            "kv_lora_rank": kv_lora_rank,
            "rope_head_dim": rope_head_dim,
            "nope_head_dim": nope_dim,
            "v_head_dim": rope_head_dim,  # typically aligned with rope
            "rope_base": 10000,
        },
        "misc": {
            "init_weight_std": 0.02,
            "rms_norm_eps": 1e-6,
            "device": "cuda",
        },
        "model": {
            "d_model": d_model,
            "nheads": nheads,
            "num_layers": num_layers,
            "max_position_embeddings": 512,
            "dropout": dropout,
            "vocab_size": 50257,
        },
        "moe": {
            "num_shared_experts": 1,
            "num_routed_experts": 2**moe_routers_exponent,
            "moe_hidden_dimension": int(moe_ratio * d_model),
            "mlp_hidden_dimension": int(moe_ratio * d_model),
            "topk": 2,
            "expert_load_balance_factor": 0.01,
            "topk_norm_epsilon": 1e-9,
            "normalized_moe_gates": True,
            "first_k_dense_replace": random.randint(a=0, b=num_layers),
            "disable_moe": False,
        },
        "rope": {
            "head_dim": rope_head_dim,
            "base": 10000,
        }
    }
    return config


def main():
    for d_model in D_MODELS:
        for nheads in HEADS:
            kv_capacity = d_model // nheads
            kv_split_options = split_kv_capacity(kv_capacity)

            for num_layers, moe_ratio, (kv_lora_rank, rope_head_dim) in itertools.product(
                LAYERS, MOE_MLP_RATIO, kv_split_options
            ):
                for dropout in DROPOUTS:
                    for nope in NOPE_DIM_OPTIONS(rope_head_dim):
                        config = generate_config(
                            d_model=d_model,
                            nheads=nheads,
                            num_layers=num_layers,
                            moe_ratio=moe_ratio,
                            kv_lora_rank=kv_lora_rank,
                            rope_head_dim=rope_head_dim,
                            nope_dim=nope,
                            dropout=dropout
                        )
                        name = (
                            f"d{d_model}_h{nheads}_l{num_layers}"
                            f"_moer{int(moe_ratio * 100)}"
                            f"_kv{kv_lora_rank}_rope{rope_head_dim}"
                            f"_nope{nope}_dropout{dropout}.yaml"
                        )
                        if random.random() < RATIO_OF_CONFIGS_TO_KEEP:
                            with open(CONFIG_DIR / name, "w") as f:
                                yaml.dump(config, f)
                            print(f"Saved: {name}")

if __name__ == "__main__":
    main()