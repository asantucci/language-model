import json
import torch
import pytest
from train.shared import get_model, configure_optimizers, configure_scheduler
from types import SimpleNamespace
from config.training_args import TrainingArgs

def test_single_train_step():
    with open("tests/fixtures/configs/training_args.json", "r") as f:
        raw_args = json.load(f)
    training_args = TrainingArgs(**raw_args)
    model = get_model(mode="pretrain", model_config_path=training_args.model_config_path, device=training_args.device)
    optimizer = configure_optimizers(model, training_args)
    scheduler = configure_scheduler(optimizer, training_args)

    # Simulate synthetic batch
    batch_size, seq_len = training_args.batch_size, training_args.seq_len
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
