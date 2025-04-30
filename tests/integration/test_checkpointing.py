import json
from omegaconf import OmegaConf
import os
from types import SimpleNamespace
import torch
import tempfile
from config.training_args import TrainingArgs
from train.shared import get_model, configure_optimizers
from tests.conftest import dummy_args
from typing import Optional

import pytest
import shutil
import os

def test_checkpoint_save_load(dummy_args):
    train_cfg = OmegaConf.load("tests/fixtures/configs/train.yaml")
    model_cfg = OmegaConf.load("tests/fixtures/configs/pretrain.yaml")
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

    model = get_model(mode="pretrain", model_cfg=cfg, device="cpu")
    optimizer = configure_optimizers(model, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": 42,
            "training_args": {},
        }, ckpt_path)

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        assert checkpoint["iter_num"] == 42
