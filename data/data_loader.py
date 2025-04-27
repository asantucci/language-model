import numpy as np
import os
import random
import torch

from datasets import load_dataset

def load_shard(file_path):
    """
    Loads a single .npy shard from disk and returns as a torch tensor.
    """
    array = np.load(file_path).astype(np.int64)  # Uses int64 for compat. with CE-Loss.
    return torch.from_numpy(array)

class DataLoader:
    def __init__(self, data_dir, batch_size, seq_len, split, device="cpu", shuffle=True):
        """
        Parameters:
            data_dir (str): Path to directory containing dataset shards.
            batch_size (int): Number of sequences per batch.
            seq_len (int): Length of each input sequence.
            split (str): 'train' or 'val'.
            device (str): 'cpu' or 'cuda'.
            shuffle (bool): Whether to shuffle shards on load/reset.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.device = device
        self.shuffle = shuffle

        all_shards = os.listdir(data_dir)
        self.shards = [f for f in all_shards if split in f]

        if shuffle:
            random.shuffle(self.shards)

        self.current_shard = 0
        self.current_pos = 0
        self.tokens = None
        self.reset_status()

    def reset_status(self):
        """Resets a (potentially shuffled) shard."""
        self.current_shard = 0
        self.current_pos = 0
        if self.shuffle:
            random.shuffle(self.shards)
        shard_path = os.path.join(self.data_dir, self.shards[self.current_shard])
        self.tokens = load_shard(shard_path)

    def next_batch(self):
        """
        Returns the next batch of input-output pairs (x, y), where:
            x = current tokens, and
            y = next tokens (shifted by 1).
        """
        span = self.batch_size * self.seq_len + 1
        batch_tokens = self.tokens[self.current_pos:self.current_pos + span]

        # Creates new tensors with a copy of data, but reshaped.
        x = batch_tokens[:-1].view(self.batch_size, self.seq_len)
        y = batch_tokens[1:].view(self.batch_size, self.seq_len)

        self.current_pos += self.batch_size * self.seq_len

        if self.current_pos + span > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_pos = 0
            shard_path = os.path.join(self.data_dir, self.shards[self.current_shard])
            self.tokens = load_shard(shard_path)

        return x.to(self.device), y.to(self.device)


class StreamingPretrainBatchDataset:
    def __init__(
            self, pth: str, tokenizer, block_size: int, batch_size: int, name=None, split="train"
        ):
        self.block_size = block_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.dataset = load_dataset(pth, name, split=split, streaming=True, trust_remote_code=True)
        self.iterator = iter(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        batch_input_ids = []
        batch_labels = []
        while len(batch_input_ids) < self.batch_size:
            tokens = []
            while len(tokens) < self.block_size + 1:
                try:
                    item = next(self.iterator)
                except StopIteration:
                    self.iterator = iter(self.dataset)
                    item = next(self.iterator)
                tokens.extend(self.tokenizer(item["text"], return_attention_mask=False)["input_ids"])

            tokens = tokens[: self.block_size + 1]
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)

        input_ids = torch.stack(batch_input_ids)   # shape: [B, T]
        labels = torch.stack(batch_labels)         # shape: [B, T]
        return {"input_ids": input_ids, "labels": labels}