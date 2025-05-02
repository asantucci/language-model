"""Trains a real Language Model (35M) on a very small batch of real data.

We obtained the following results:
https://wandb.ai/asantucci-stanford-university/tiny-deepseek-test/runs/6dkgnxc4.
The configuration that was used (for record-keeping, we dump the raw values here):

kv_cache:
  use_kv_cache: false
  q_lora_rank: null
  kv_lora_rank: 64  # KV_LORA_rank + rope_head_dim == d_model // nheads == head_dim.
  rope_head_dim: 64
  nope_head_dim: 32
  v_head_dim: 64
  rope_base: 10000

misc:
  init_weight_std: 0.02
  rms_norm_eps: 1e-6
  device: cuda

model:
  d_model: 256
  nheads: 2
  num_layers: 2
  max_position_embeddings: 512
  dropout: 0.1
  vocab_size: 50257

moe:
  num_shared_experts: 1
  num_routed_experts: 4
  moe_hidden_dimension: 2048
  mlp_hidden_dimension: 2048
  topk: 2
  expert_load_balance_factor: 0.01
  topk_norm_epsilon: 1e-9
  normalized_moe_gates: true
  first_k_dense_replace: 1
  disable_moe: false

rope:
  head_dim: 64
  base: 10000

checkpoint:
  resume: ""
  out_dir: "./checkpoints"
  checkpoint_path: "pretrain_ckpt"
  save_interval: 5000

data:
  hf_dataset_name: "roneneldan/TinyStories"
  hf_dataset_dir: ""

optim:
  learning_rate: 2e-2
  min_learning_rate: 1e-3
  scheduler_type: "one_cycle"
  lr_decay_iters: 25000
  pct_warmup: 0.1
  adamw_beta1: 0.9
  adamw_beta2: 0.95
  adamw_weight_decay: 0.01
  adamw_use_fused: true
  use_eight_bit_optimizer: false
  grad_clip: 1.0

tokenizer:
  pretrained_tokenizer_name: "openai-community/gpt2"
  pretrained_tokenizer_max_length: 2048

train:
  batch_size: 2
  seq_len: 128
  max_train_steps: 1000000
  gradient_accumulation_steps: 4
  eval_interval: 500
  eval_iters: 10
  generate_interval: 500
  dtype: "bfloat16"
  device: "cuda"

wandb:
  wandb_project: tiny-deepseek-test
  wandb_run_name: pretrain_tiny
  wandb_log: true
  log_interval: 10
"""
import torch
from omegaconf import OmegaConf
import wandb

from datasets import load_dataset
from train import shared

def make_fixed_batch_loader(tokenizer, n_examples=8000, block_size=256, batch_size=8):
    """Creates a very small batch of `n_examples`, which we cycle over during training."""
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    input_ids_list = []
    labels_list = []

    iterator = iter(dataset)
    while len(input_ids_list) < n_examples:
        item = next(iterator)
        tokens = tokenizer(item["text"], return_attention_mask=False, add_special_tokens=False)["input_ids"]
        if len(tokens) < 2:
            continue
        tokens.append(tokenizer.eos_token_id)
        tokens = tokens[:block_size + 1]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        if len(input_ids) < block_size:
            pad_len = block_size - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])  # mask for loss

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    # Group into batches
    assert len(input_ids_list) % batch_size == 0, "n_examples must be divisible by batch_size"
    n_batches = len(input_ids_list) // batch_size
    batch_list = []

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch = {
            "input_ids": torch.stack(input_ids_list[start:end]),
            "labels": torch.stack(labels_list[start:end]),
        }
        batch_list.append(batch)

    # Loader cycles through all batches
    class Loader:
        def __init__(self, batch_list):
            self.batches = batch_list
            self.i = 0

        def __iter__(self): return self
        def __next__(self): return self.safe_next()[0]
        def safe_next(self):
            batch = self.batches[self.i % len(self.batches)]
            self.i += 1
            return batch, False

    return Loader(batch_list)

shared.get_dataloader = lambda cfg, split, tokenizer: make_fixed_batch_loader(tokenizer=tokenizer, n_examples=4096, block_size=cfg.seq_len, )
model_cfg = OmegaConf.load("config/model/tiny.yaml")
train_cfg = OmegaConf.load("config/train/tiny.yaml")
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

wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=dict(cfg), reinit=True)
shared.train_loop(cfg, mode='pretrain', wandb_logger=wandb_run)
