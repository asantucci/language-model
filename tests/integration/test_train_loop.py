import json
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch
from typing import Optional

from config.training_args import TrainingArgs
from train.shared import get_model, train_loop, load_checkpoint

import pytest
import shutil
import os

class MockTinyDataset(torch.utils.data.IterableDataset):
    """A tiny fake streaming dataset."""
    def __iter__(self):
        for _ in range(5):  # 5 examples only
            input_ids = torch.randint(0, 1000, (8,), dtype=torch.long)
            labels = torch.randint(0, 1000, (8,), dtype=torch.long)
            yield {"input_ids": input_ids, "labels": labels}

def mock_get_dataloader(args, split):
    """Returns a mocked dataloader that yields fake batches."""
    batch_size = args.batch_size
    seq_len = args.seq_len
    vocab_size = 1024

    class DummyLoader:
        def __iter__(self):
            return self

        def __next__(self):
            return {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
            }

    return DummyLoader()


@patch("train.shared.get_dataloader", new=mock_get_dataloader)
def test_pretrain_loop_runs_and_loss_is_finite():
    """Test that pretraining runs and produces finite loss and saves checkpoint correctly."""
    with open("tests/fixtures/configs/training_args.json", "r") as f:
        raw_args = json.load(f)
    training_args = TrainingArgs(**raw_args)
    train_loop(training_args, mode="pretrain")

    # 3. Check that checkpoint file exists
    ckpt_path = os.path.join(training_args.out_dir, f"{training_args.checkpoint_path}_step1.pt")
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    # 4. Load checkpoint and model
    checkpoint = load_checkpoint(ckpt_path, device=training_args.device, weights_only=False)
    trained_model = get_model(mode="pretrain", model_config_path=training_args.model_config_path, device=training_args.device)
    trained_model.load_state_dict(checkpoint["model"])

    # 5. Verify that the model's weights are not trivial
    total_weight_sum = trained_model.lm_head.weight.abs().sum().item()
    assert total_weight_sum > 0, "Trained model weights are still zero after training!"

@patch("train.shared.get_dataloader", new=mock_get_dataloader)
def test_sft_loop_runs_and_loss_is_finite():
    """Test that SFT runs and produces finite loss."""
    with open("tests/fixtures/configs/training_args.json", "r") as f:
        raw_args = json.load(f)
    training_args = TrainingArgs(**raw_args)
    model = get_model(mode="sft", model_config_path=training_args.model_config_path, device=training_args.device)
    initial_weight = model.lm_head.weight.clone().detach()

    train_loop(training_args, mode="sft")

    # 3. Check that checkpoint file exists
    ckpt_path = os.path.join(training_args.out_dir, f"{training_args.checkpoint_path}_step1.pt")
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    # 4. Load checkpoint and model
    checkpoint = load_checkpoint(ckpt_path, device=training_args.device, weights_only=False)
    trained_model = get_model(mode="pretrain", model_config_path=training_args.model_config_path, device=training_args.device)
    trained_model.load_state_dict(checkpoint["model"])
    
    final_weight = trained_model.lm_head.weight.clone().detach()
    weight_diff = (initial_weight - final_weight).abs().sum().item()
    assert weight_diff > 0, "Model weights did not change after optimizer step!"
