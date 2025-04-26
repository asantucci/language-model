import os
from types import SimpleNamespace
import torch
import tempfile
from train.shared import get_model, configure_optimizers

def test_checkpoint_save_load():
    model = get_model(device="cpu", mode="pretrain")
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
