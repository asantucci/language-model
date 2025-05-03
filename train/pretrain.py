import argparse
from omegaconf import OmegaConf
from train.shared import train_loop, LinearSlopeBasedEarlyStopper
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, default="config/model/medium.yaml")
    parser.add_argument("--train-config", type=str, default="config/train/base_pretrain.yaml")
    parser.add_argument("--early-stopping", type=bool, default=False)
    parser.add_argument("--early-stopping-config", type=str, default="config/early_stopping/stop.yaml")
    args = parser.parse_args()

    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.train_config)
    cfg = OmegaConf.merge(
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
    wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=dict(cfg), reinit=True)
    if args.early_stopping:
        early_stopper_cfg = OmegaConf.load(args.early_stopping_config)
        monitor = LinearSlopeBasedEarlyStopper(
            window_size=early_stopper_cfg.window_size,
            slope_history_window=early_stopper_cfg.slope_history_window,
            min_steps_before_check=early_stopper_cfg.min_steps_before_check,
            ema_beta=early_stopper_cfg.ema_beta,
            slope_mean_threshold=early_stopper_cfg.slope_mean_threshold,
            slope_magnitude_threshold=early_stopper_cfg.slope_magnitude_threshold,
            slope_fraction_threshold=early_stopper_cfg.slope_fraction_threshold,
            wandb_logger=wandb_run
        )
    else:
        monitor = None
    train_loop(cfg, mode="pretrain", loss_monitor=monitor, wandb_logger=wandb_run)

if __name__ == "__main__":
    main()
