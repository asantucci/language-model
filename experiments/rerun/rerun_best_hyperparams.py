"""Reruns the best hyperparameters from a sweep.

Example usage:
python rerun/rerun_best_hyperparams.py \
  --best-yaml rerun/rerun_configs/top5_best_configs.yaml \
  --save-config-dir rerun/rerun_configs \
  --max-train-steps 100000
"""
import os
import argparse
import yaml
from omegaconf import OmegaConf

from train.shared import train_loop

def load_best_configs(best_yaml_path):
    """
    Loads best hyperparameter configs from a YAML file.

    Args:
        best_yaml_path (str): Path to YAML containing list of best configs.

    Returns:
        list of dict: List of best hyperparameter settings.
    """
    with open(best_yaml_path, "r") as f:
        return yaml.safe_load(f)

def generate_rerun_cfg(base_train_cfg, base_model_cfg, hyperparams, max_train_steps=100_000):
    """
    Generates final merged configuration for a rerun.

    Args:
        base_train_cfg (OmegaConf): Base training configuration.
        base_model_cfg (OmegaConf): Base model configuration.
        hyperparams (dict): Specific hyperparameters to override.
        max_train_steps (int): Number of steps for full training.

    Returns:
        OmegaConf: Final merged config.
    """
    base_merged_cfg = OmegaConf.merge(
        base_model_cfg.kv_cache,
        base_model_cfg.misc,
        base_model_cfg.model,
        base_model_cfg.moe,
        {"rope": base_model_cfg.rope},
        base_train_cfg.checkpoint,
        base_train_cfg.data,
        base_train_cfg.optim,
        base_train_cfg.tokenizer,
        base_train_cfg.train,
        base_train_cfg.wandb,
    )

    # Apply hyperparameter overrides
    final_cfg = OmegaConf.merge(base_merged_cfg, hyperparams)

    # Final adjustments
    final_cfg.max_train_steps = max_train_steps
    final_cfg.checkpoint.resume = ""  # No resume — fresh training
    final_cfg.wandb_run_name = f"rerun_lr_{hyperparams['learning_rate']}_warmup_{hyperparams['pct_warmup']}"

    return final_cfg

def save_rerun_cfg(cfg, save_dir, run_name):
    """
    Saves the rerun config for reproducibility.

    Args:
        cfg (OmegaConf): Config to save.
        save_dir (str): Directory to save into.
        run_name (str): Used to name the config file.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{run_name}.yaml")
    with open(save_path, "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

def main():
    parser = argparse.ArgumentParser(description="Rerun best hyperparameters for full training.")
    parser.add_argument("--best-yaml", type=str, required=True, help="Path to YAML with best hyperparams (from sweep analysis).")
    parser.add_argument("--save-config-dir", type=str, default="rerun/rerun_configs", help="Directory to save rerun configs.")
    parser.add_argument("--max-train-steps", type=int, default=100_000, help="Max training steps for reruns.")
    args = parser.parse_args()

    base_train_cfg = OmegaConf.load("config/train/base_pretrain.yaml")
    base_model_cfg = OmegaConf.load("config/model/medium.yaml")

    best_configs = load_best_configs(args.best_yaml)

    for idx, hyperparams in enumerate(best_configs):
        print(f"Rerunning best config #{idx+1}: {hyperparams}")

        final_cfg = generate_rerun_cfg(base_train_cfg, base_model_cfg, hyperparams, max_train_steps=args.max_train_steps)

        save_rerun_cfg(final_cfg, args.save_config_dir, final_cfg.wandb_run_name)

        train_loop(final_cfg, mode="pretrain")

if __name__ == "__main__":
    main()
