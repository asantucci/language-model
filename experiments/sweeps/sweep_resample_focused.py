"""
Example usage:
uv run python3 experiments/sweeps/sweep_resample_focused.py \
    --best-yaml=/tmp/best_configs \
    --n-points-per-best=3 \
    --wandb-project=deepseek_focused_sweep \
    --max-train-steps=30000 \
    --early-stopping=true \
    --patience=3 \
    --relative-flat-tolerance=0.01 \
    --min-steps-before-check=1000
"""
import numpy as np
import argparse
import yaml
from omegaconf import OmegaConf

from train.shared import TrainingLossMonitor, train_loop

def sample_log_uniform(low, high):
    log_low = np.log(low)
    log_high = np.log(high)
    return np.exp(np.random.uniform(log_low, log_high))

def sample_uniform(low, high):
    return np.random.uniform(low, high)

def focused_generate_sweep_points(base_lr, base_warmup, n=5, lr_factor=2.0, warmup_factor=1.5):
    """
    Samples new sweep points around a base learning rate and warmup.

    Args:
        base_lr (float): Base learning rate to perturb around.
        base_warmup (float): Base warmup pct to perturb around.
        n (int): Number of sweep points.
        lr_factor (float): Multiplier range for learning rate.
        warmup_factor (float): Multiplier range for warmup pct.

    Returns:
        list of dict: List of new hyperparameter configs.
    """
    points = []
    for _ in range(n):
        lr = sample_log_uniform(0.5 * base_lr, lr_factor * base_lr)
        warmup = np.clip(sample_uniform(0.5 * base_warmup, warmup_factor * base_warmup), 0.02, 0.3)
        points.append({
            "learning_rate": float(lr),
            "pct_warmup": float(warmup),
        })
    return points

def main():
    parser = argparse.ArgumentParser(description="Focused re-sweep around best hyperparameters.")
    parser.add_argument("--best-yaml", type=str, required=True, help="Path to YAML with best hyperparams (from sweep analysis).")
    parser.add_argument("--n-points-per-best", type=int, default=5, help="Number of samples to generate around each best config.")
    parser.add_argument("--wandb-project", type=str, help="Title of the Project to create in WandB.ai")
    parser.add_argument("--max-train-steps", type=int, help="# of training steps to execute for each trial.")
    parser.add_argument("--early-stopping", type=bool, help="Whether to use a TrainingLossMonitor to consider an Early Stopping heuristic.")
    parser.add_argument("--patience", type=int, help="# of logged steps to consider using TrainingLossMonitor.")
    parser.add_argument("--relative-flat-tolerance", type=float, help="The absolute tolerance in adjacent log-steps to be considered 'no progress'.")
    parser.add_argument("--min-steps-before-check", type=int, help="Minimum # of training steps to execute before considering Early Stopping.")
    args = parser.parse_args()

    base_train_cfg = OmegaConf.load("config/train/base_pretrain.yaml")
    model_cfg = OmegaConf.load("config/model/medium.yaml")

    with open(args.best_yaml, "r") as f:
        best_points = yaml.safe_load(f)

    for idx, best_cfg in enumerate(best_points):
        print(f"Generating focused sweep around best config #{idx+1}: {best_cfg}")

        sweep_points = focused_generate_sweep_points(
            base_lr=best_cfg["learning_rate"],
            base_warmup=best_cfg["pct_warmup"],
            n=args.n_points_per_best,
        )

        for sweep_idx, sweep_cfg in enumerate(sweep_points):
            base_merged_cfg = OmegaConf.merge(
                model_cfg.kv_cache,
                model_cfg.misc,
                model_cfg.model,
                model_cfg.moe,
                {"rope": model_cfg.rope},
                base_train_cfg.checkpoint,
                base_train_cfg.data,
                base_train_cfg.optim,
                base_train_cfg.tokenizer,
                base_train_cfg.train,
                base_train_cfg.wandb,
            )

            final_cfg = OmegaConf.merge(base_merged_cfg, sweep_cfg)
            final_cfg.max_train_steps = args.max_train_steps
            final_cfg.resume = ""
            final_cfg.wandb_run_name = f"focused_sweep{idx}_{sweep_idx}_lr{final_cfg.learning_rate:.1e}_wu{final_cfg.pct_warmup:.2f}"
            final_cfg.wandb_project = "deepseek_focused_sweep"

            # Create a monitor for this sweep run
            if args.early_stopping:
                monitor = TrainingLossMonitor(
                    patience=args.patience,
                    relative_flat_tolerance=args.relative_flat_tolerance,
                    min_steps_before_check=args.min_steps_before_check,
                )
            else:
                monitor = None
            train_loop(final_cfg, mode="pretrain", loss_monitor=monitor)

if __name__ == "__main__":
    main()
