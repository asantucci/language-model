# tests/integration/test_train_tiny_dataset.py

import pytest
import torch
from types import SimpleNamespace
from datasets import Dataset
from transformers import AutoTokenizer

from train.shared import get_model, configure_optimizers, configure_scheduler
import train.shared as shared

class MockStreamingPretrainBatchDataset:
    def __init__(self, block_size, batch_size):
        self.block_size = block_size
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        # Create a fake batch immediately
        input_ids = torch.randint(0, 100, (self.batch_size, self.block_size))
        labels = torch.randint(0, 100, (self.batch_size, self.block_size))
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

@pytest.fixture
def dummy_args(tmp_path):
    return SimpleNamespace(
        learning_rate=1e-4,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        adamw_weight_decay=0.01,
        adamw_use_fused=False,
        grad_clip=1,
        decay_lr=True,
        scheduler_type="linear",
        lr_decay_iters=10,
        min_learning_rate=1e-5,
        use_eight_bit_optimizer=False,
        device="cpu",
        batch_size=2,
        seq_len=8,
        max_train_steps=5,
        gradient_accumulation_steps=1,
        wandb_log=False,
        wandb_project="dummy",
        wandb_run_name="dummy_run",
        save_interval=1000,
        out_dir=str(tmp_path),  # important: something writable
        checkpoint_path="dummy",
        resume=None,
        hf_dataset_name="wikitext",
        hf_dataset_dir="wikitext-2-raw-v1",
        eval_interval=1000,
        eval_iters=1,
        dtype="float32",
        log_interval=1,
    )

def test_train_loop_fast(dummy_args, monkeypatch):
    """Integration test for train_loop() with a mocked fast dataloader."""

    # Keep unused arg since we're going to patch the method back into the module.
    def mock_get_dataloader(args, split):
        return MockStreamingPretrainBatchDataset(
            block_size=args.seq_len,
            batch_size=args.batch_size,
        )

    monkeypatch.setattr(shared, "get_dataloader", mock_get_dataloader)

    shared.train_loop(dummy_args, mode="pretrain")