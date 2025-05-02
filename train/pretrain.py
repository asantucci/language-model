import argparse
from omegaconf import OmegaConf
from train.shared import train_loop
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, default="config/model/medium.yaml")
    parser.add_argument("--train-config", type=str, default="config/train/base_pretrain.yaml")
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
    train_loop(cfg, mode="pretrain", wandb_logger=wandb_run)

if __name__ == "__main__":
    main()
