import pytest
import torch
from types import SimpleNamespace
from train.shared import get_model, configure_scheduler, configure_optimizers, get_dataloader, load_checkpoint

@pytest.fixture(scope="session")
def dummy_args():
    return SimpleNamespace(
        learning_rate=1e-4,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        adamw_weight_decay=0.01,
        adamw_use_fused=False,
        decay_lr=True,
        scheduler_type="cosine",
        lr_decay_iters=1000,
        min_learning_rate=1e-5,
        use_eight_bit_optimizer=False,
        hf_dataset_name="wikitext",
        hf_dataset_dir="wikitext-2-raw-v1",
        seq_len=128,
        batch_size=4,
        device="cpu",
    )

@pytest.fixture(scope="session")
def model(dummy_args):
    """Instantiate once for all tests."""
    return get_model(device=dummy_args.device, mode="pretrain")

@pytest.fixture()
def dummy_optimizer(model, dummy_args):
    """Use a real optimizer for scheduler tests."""
    return configure_optimizers(model, dummy_args)

def test_get_model_instantiates_correctly(model):
    assert model is not None
    assert hasattr(model, "forward")

def test_configure_scheduler_cosine(dummy_optimizer, dummy_args):
    scheduler = configure_scheduler(dummy_optimizer, dummy_args)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

def test_configure_optimizers(dummy_optimizer):
    assert isinstance(dummy_optimizer, torch.optim.AdamW)

def test_get_dataloader(dummy_args):
    dataloader = get_dataloader(dummy_args, split="train")
    assert hasattr(dataloader, "__iter__")

def test_load_checkpoint_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nonexistent.pt", device="cpu")
