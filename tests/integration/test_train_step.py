import json
from omegaconf import OmegaConf
import torch
import pytest
from train.shared import get_model, configure_optimizers, configure_scheduler
from types import SimpleNamespace
from config.training_args import TrainingArgs

def test_single_train_step():
    train_cfg = OmegaConf.load("tests/fixtures/configs/train.yaml")
    model_cfg = OmegaConf.load("tests/fixtures/configs/sft.yaml")
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
    model = get_model(mode="sft", model_cfg=cfg, device=cfg.device)
    optimizer = configure_optimizers(model, cfg)
    scheduler = configure_scheduler(optimizer, cfg)

    # Simulate synthetic batch
    batch_size, seq_len = cfg.batch_size, cfg.seq_len
    vocab_size = model.config.vocab_size
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    model.train()
    optimizer.zero_grad()

    _, loss, _ = model(inputs, targets)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # Optionally check parameters changed
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
