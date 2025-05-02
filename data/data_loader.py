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
    def __init__(self, pth, tokenizer, block_size, batch_size, name=None, split="train"):
        self.block_size = block_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.dataset = load_dataset(
            pth, name, split=split, streaming=True, trust_remote_code=True
        ).shuffle(buffer_size=10000, seed=42)
        self.iterator = iter(self.dataset)

    def __iter__(self): return self

    def safe_next(self):
        try:
            return next(self), False
        except StopIteration:
            self.iterator = iter(self.dataset)
            return next(self), True

    def __next__(self):
        input_ids_list = []
        labels_list = []

        while len(input_ids_list) < self.batch_size:
            item = next(self.iterator)
            text = item.get("text", "")
            tokens = self.tokenizer(
                text, return_attention_mask=False, add_special_tokens=False
            )["input_ids"]

            if len(tokens) < 2:
                continue

            tokens.append(self.tokenizer.eos_token_id)
            tokens = tokens[: self.block_size + 1]

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)

            # Pad input_ids and labels to full block_size
            pad_len = self.block_size - len(input_ids)
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id)])
                labels = torch.cat([labels, torch.full((pad_len,), -100)])

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
        }
