import json
import pytest
import torch
from types import SimpleNamespace
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Optional

from config.training_args import TrainingArgs
from train.shared import get_model, configure_optimizers, configure_scheduler
import train.shared as shared
from tests.conftest import dummy_args

class MockStreamingPretrainBatchDataset:
    def __init__(self, block_size, batch_size, vocab_size):
        self.block_size = block_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size

    def __iter__(self):
        return self

    def __next__(self):
        # Create a fake batch immediately
        input_ids = torch.randint(0, self.vocab_size-1, (self.batch_size, self.block_size))
        labels = torch.randint(0, self.vocab_size-1, (self.batch_size, self.block_size))
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

def mock_get_dataloader(args, split):
    class DummyLoader:
        def __iter__(self):
            return self

        def __next__(self):
            vocab_size = 1024
            batch_size = args.batch_size
            seq_len = args.seq_len

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))

            return {"input_ids": input_ids, "labels": labels}
    return DummyLoader()

def test_train_loop_fast(dummy_args, monkeypatch):
    """Integration test for train_loop() with a mocked fast dataloader."""
    monkeypatch.setattr(shared, "get_dataloader", mock_get_dataloader)
    with open("tests/fixtures/configs/training_args.json", "r") as f:
        raw_args = json.load(f)
    training_args = TrainingArgs(**raw_args)
    shared.train_loop(training_args, mode="pretrain")