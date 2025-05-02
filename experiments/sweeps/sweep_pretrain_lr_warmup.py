import numpy as np
from omegaconf import OmegaConf
import wandb

from train.shared import LinearSlopeBasedEarlyStopper, train_loop

def sample_log_uniform(low, high):
    log_low = np.log(low)
    log_high = np.log(high)
    return np.exp(np.random.uniform(log_low, log_high))

def sample_uniform(low, high):
    return np.random.uniform(low, high)

def generate_sweep_points(n=16):
    points = []
    for _ in range(n):
        lr = sample_log_uniform(2e-3, 1e-2)
        warmup = sample_uniform(0.01, 0.5)
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
    base_merged_cfg.max_train_steps = 25_000

    for sweep_override in sweep_cfgs:
        # Now apply the sweep overrides (learning_rate + pct_warmup)
        final_cfg = OmegaConf.merge(base_merged_cfg, sweep_override)

        # Dynamically adjust WandB run name to track different sweeps
        final_cfg.wandb_run_name = (
            f"lr_{sweep_override['learning_rate']}_warmup_{sweep_override['pct_warmup']}"
        )

        final_cfg.wandb_project = "deepseek_sweep"        
        wandb_logger = wandb.init(project=final_cfg.wandb_project, name=final_cfg.wandb_run_name, config=dict(final_cfg), reinit="finish_previous")
        # Create a monitor for this sweep run
        monitor = LinearSlopeBasedEarlyStopper(
            window_size=250,
            slope_history_window=250,
            min_steps_before_check=1000,
            ema_beta=0.9,
            slope_mean_threshold=0.001,
            slope_magnitude_threshold=0.0005,
            slope_fraction_threshold=0.1,
            wandb_logger=wandb_logger
        )
        print(f"Launching sweep run: {final_cfg.wandb_run_name}")
        train_loop(final_cfg, mode="pretrain", loss_monitor=monitor, wandb_logger=wandb_logger)

if __name__ == "__main__":
    main()
