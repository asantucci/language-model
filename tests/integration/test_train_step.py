# tests/integration/test_train_step.py

import torch
import pytest
from train.shared import get_model, configure_optimizers, configure_scheduler
from types import SimpleNamespace

@pytest.fixture
def dummy_args():
    return SimpleNamespace(
        learning_rate=1e-4,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        adamw_weight_decay=0.01,
        adamw_use_fused=False,
        decay_lr=True,
        scheduler_type="linear",
        lr_decay_iters=10,
        min_learning_rate=1e-5,
        use_eight_bit_optimizer=False,
        hf_dataset_name="wikitext",
        hf_dataset_dir="wikitext-2-raw-v1",
        seq_len=8,
        batch_size=2,
        device="cpu",
    )

def test_single_train_step(dummy_args):
    model = get_model(device=dummy_args.device, mode="pretrain")
    optimizer = configure_optimizers(model, dummy_args)
    scheduler = configure_scheduler(optimizer, dummy_args)

    # Simulate synthetic batch
    batch_size, seq_len = dummy_args.batch_size, dummy_args.seq_len
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
