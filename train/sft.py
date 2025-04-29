import argparse
from omegaconf import OmegaConf
from train.shared import train_loop

def main():
    parser = argparse.ArgumentParser(description="DeepSeek Supervised Fine-Tuning (SFT) Script")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--train-config", type=str, required=True)
    args = parser.parse_args()

    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.train_config)
    cfg = OmegaConf.merge(
        model_cfg.kv_cache,
        model_cfg.misc,
        model_cfg.model,
        model_cfg.moe,
        train_cfg.data,
        train_cfg.optim,
        train_cfg.tokenizer,
        train_cfg.train,
        train_cfg.wandb,
    )

    train_loop(cfg, mode="sft")

if __name__ == "__main__":
    main()
