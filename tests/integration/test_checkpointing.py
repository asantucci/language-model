import json
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
    with open("tests/fixtures/configs/training_args.json", "r") as f:
        raw_args = json.load(f)
    training_args = TrainingArgs(**raw_args)
    dummy_args.model_config_path = 'tests/fixtures/configs/pretrain.json'
    model = get_model(mode="pretrain", model_config_path=dummy_args.model_config_path, device=training_args.device)
    optimizer = configure_optimizers(model, SimpleNamespace(
        learning_rate=1e-4,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        adamw_weight_decay=0.01,
        adamw_use_fused=False,
        use_eight_bit_optimizer=False,
    ))

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
