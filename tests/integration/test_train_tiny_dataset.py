import json
from omegaconf import OmegaConf
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
            batch, _ = self.safe_next()
            return batch

        def safe_next(self):
            vocab_size = 1024
            batch_size = args.batch_size
            seq_len = args.seq_len

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))

            return {"input_ids": input_ids, "labels": labels}, False
    return DummyLoader()

def test_train_loop_fast(dummy_args, monkeypatch):
    """Integration test for train_loop() with a mocked fast dataloader."""
    monkeypatch.setattr(shared, "get_dataloader", mock_get_dataloader)
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
    shared.train_loop(cfg, mode="pretrain")