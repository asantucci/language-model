import torch
from train.shared import train_loop
from omegaconf import OmegaConf

def mock_overfit_dataloader(cfg, split):
    batch_size = cfg.train.batch_size
    seq_len = cfg.train.seq_len
    vocab_size = cfg.model.vocab_size

    fixed_input = torch.randint(5, vocab_size - 5, (batch_size, seq_len))
    fixed_labels = fixed_input.clone()

    class DummyLoader:
        def __iter__(self):
            return self

        def __next__(self):
            return {"input_ids": fixed_input, "labels": fixed_labels}

        def safe_next(self):
            return {"input_ids": fixed_input, "labels": fixed_labels}, False

    return DummyLoader()


def mock_toy_dataloader(cfg, split):
    batch_size = cfg.batch_size
    seq_len = cfg.seq_len
    vocab_size = cfg.vocab_size

    # Make rolling sequences of length (seq_len + 1)
    train_dataset = [list(range(i, i + seq_len + 1)) for i in range(vocab_size - seq_len - 1)]
    val_dataset = [[t + 100 for t in seq] for seq in train_dataset[:10]]  # small shifted OOD slice
    dataset = train_dataset if split == "train" else val_dataset

    class ToyLoader:
        def __init__(self):
            self.idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            return self.safe_next()[0]

        def safe_next(self):
            batch_input_ids = []
            batch_labels = []
            for _ in range(batch_size):
                seq = dataset[self.idx % len(dataset)]
                self.idx += 1
                input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                labels = torch.tensor(seq[1:], dtype=torch.long)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)

            return {
                "input_ids": torch.stack(batch_input_ids),
                "labels": torch.stack(batch_labels),
            }, False

    return ToyLoader()

class MockWandbLogger:
    def __init__(self):
        self.losses = []
        self.val_losses = []

    def log(self, metrics: dict):
        if "Train Loss" in metrics:
            self.losses.append(metrics["Train Loss"])
        if "Validation Loss" in metrics:
            self.val_losses.append(metrics["Validation Loss"])


def test_loss_decreases_on_counting(monkeypatch):
    train_cfg = OmegaConf.load("tests/fixtures/configs/train.yaml")
    model_cfg = OmegaConf.load("tests/fixtures/configs/tiny.yaml")
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

    # Patch your dataloader
    monkeypatch.setattr("train.shared.get_dataloader", mock_toy_dataloader)
    cfg.max_train_steps=2500

    # Capture losses
    mock_logger = MockWandbLogger()
    train_loop(cfg, mode="sft", wandb_logger=mock_logger)

    assert mock_logger.losses[0] > mock_logger.losses[-1]
    assert mock_logger.val_losses[0] > mock_logger.val_losses[-1], "Validation loss did not improve"
    assert mock_logger.val_losses[-1] < 1.0, f"Final validation loss too high: {mock_logger.val_losses[-1]}"
