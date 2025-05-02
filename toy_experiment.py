from omegaconf import OmegaConf
from train.shared import train_loop, get_model
from tests.integration.test_train_learns_a_sequence import mock_toy_dataloader
import train.shared

class LiveLogger:
    def __init__(self):
        self.losses = []
        self.val_losses = []

    def log(self, metrics):
        if "Train Loss" in metrics:
            self.losses.append(metrics["Train Loss"])
        if "Validation Loss" in metrics:
            self.val_losses.append(metrics["Validation Loss"])
        print(metrics)

def toy_experiment():
    train_cfg = OmegaConf.load("tests/fixtures/configs/train.yaml")
    model_cfg = OmegaConf.load("tests/fixtures/configs/tiny.yaml")
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
    cfg.eval_interval = 5
    cfg.save_interval = 10
    cfg.generate_interval = 5
    cfg.eval_iters = 2
    cfg.max_train_steps = 1000

    logger = LiveLogger()

    # Monkeypatch directly
    train.shared.get_dataloader = mock_toy_dataloader

    train_loop(cfg, mode="sft", wandb_logger=logger)


if __name__ == "__main__":
    toy_experiment()
