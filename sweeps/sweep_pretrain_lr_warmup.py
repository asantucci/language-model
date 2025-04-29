import numpy as np
from omegaconf import OmegaConf

from train.shared import TrainingLossMonitor, train_loop

def sample_log_uniform(low, high):
    log_low = np.log(low)
    log_high = np.log(high)
    return np.exp(np.random.uniform(log_low, log_high))

def sample_uniform(low, high):
    return np.random.uniform(low, high)

def generate_sweep_points(n=5):
    points = []
    for _ in range(n):
        lr = sample_log_uniform(5e-5, 2e-3)
        warmup = sample_uniform(0.02, 0.3)
        points.append({
            "learning_rate": float(lr),
            "pct_warmup": float(warmup),
        })
    return points

def main():
    train_cfg = OmegaConf.load("config/train/base_pretrain.yaml")
    model_cfg = OmegaConf.load("config/model/medium.yaml")
    sweep_cfgs = generate_sweep_points(n=16)
    base_merged_cfg = OmegaConf.merge(
        model_cfg.kv_cache,
        model_cfg.misc,
        model_cfg.model,
        model_cfg.moe,
        {"rope": model_cfg.rope},
        train_cfg.checkpoint,
        train_cfg.data,
        train_cfg.optim,
        train_cfg.tokenizer,
        train_cfg.train,
        train_cfg.wandb,
    )

    for sweep_override in sweep_cfgs:
        # Now apply the sweep overrides (learning_rate + pct_warmup)
        final_cfg = OmegaConf.merge(base_merged_cfg, sweep_override)

        # Dynamically adjust WandB run name to track different sweeps
        final_cfg.wandb_run_name = (
            f"lr_{sweep_override['learning_rate']}_warmup_{sweep_override['pct_warmup']}"
        )

        # Create a monitor for this sweep run
        monitor = TrainingLossMonitor(
            patience=10,
            flat_tolerance=0.1,
            min_steps_before_check=1000,
        )

        print(f"🚀 Launching sweep run: {final_cfg.wandb_run_name}")
        train_loop(final_cfg, mode="pretrain", loss_monitor=monitor)

if __name__ == "__main__":
    main()
