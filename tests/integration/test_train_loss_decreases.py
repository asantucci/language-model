import torch
import pytest
from omegaconf import OmegaConf
from train.shared import train_loop, get_dataloader

@pytest.fixture
def mock_overfit_dataloader():
    def _loader(cfg, split):
        batch_size = cfg.batch_size
        seq_len = cfg.seq_len
        vocab_size = cfg.vocab_size

        fixed_input = torch.randint(5, vocab_size - 5, (batch_size, seq_len))
        fixed_labels = fixed_input.clone()

        class DummyLoader:
            def __iter__(self):
                return self

            def __next__(self):
                return {"input_ids": fixed_input, "labels": fixed_labels}

            def safe_next(self):
                return {"input_ids": fixed_input, "labels": fixed_labels}, False

        return DummyLoader()
    return _loader

class MockWandbLogger:
    def __init__(self):
        self.losses = []

    def log(self, metrics: dict):
        if "Train Loss" in metrics:
            self.losses.append(metrics["Train Loss"])

def test_loss_decreases_to_zero(monkeypatch, mock_overfit_dataloader):
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

    # Patch data loader
    monkeypatch.setattr("train.shared.get_dataloader", mock_overfit_dataloader)

    # Capture losses
    mock_logger = MockWandbLogger()

    train_loop(cfg, mode="pretrain", wandb_logger=mock_logger)

    assert mock_logger.losses[0] > mock_logger.losses[-1], "Loss did not decrease"
    assert mock_logger.losses[-1] < 1.0, f"Final loss too high: {mock_logger.losses[-1]}"
