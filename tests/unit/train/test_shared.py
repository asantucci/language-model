import json
import pytest
import torch

from config.training_args import TrainingArgs
from train.shared import (
    get_model,
    configure_scheduler,
    configure_optimizers,
    get_dataloader,
    load_checkpoint,
)

# === Fixtures ===

@pytest.fixture
def training_args():
    """Load training args from config."""
    with open("tests/fixtures/configs/training_args.json", "r") as f:
        args = json.load(f)
    return TrainingArgs(**args)

@pytest.fixture
def model(training_args):
    """Instantiate model for tests."""
    return get_model(
        mode="pretrain",
        model_config_path="tests/fixtures/configs/pretrain.json",
        device=training_args.device,
    )

@pytest.fixture
def optimizer(model, training_args):
    """Create optimizer for tests."""
    training_args.use_eight_bit_optimizer = False
    return configure_optimizers(model, training_args)

# === Tests ===

def test_get_model_instantiates_correctly(model):
    assert model is not None
    assert hasattr(model, "forward")

def test_configure_scheduler_cosine(optimizer, training_args):
    scheduler = configure_scheduler(optimizer, training_args)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

def test_configure_optimizers(optimizer):
    assert isinstance(optimizer, torch.optim.AdamW)

def test_get_dataloader(training_args):
    dataloader = get_dataloader(training_args, split="train")
    assert hasattr(dataloader, "__iter__")

def test_load_checkpoint_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nonexistent.pt", device="cpu", weights_only=False)
