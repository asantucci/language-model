"""
Training utilities for DeepSeek-style models.

Design:
- Single-GPU friendly
- Mixed-precision supported (via torch.amp.autocast)
- Lightweight (no distributed training, no DeepSpeed/FSDP)
"""
from dataclasses import dataclass
import gc
import os
import torch
import wandb
from contextlib import nullcontext
import time
from typing import Optional

import bitsandbytes as bnb
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR

from config.deepseek import DeepSeekConfig
from config.training_args import TrainingArgs
from model.deepseek import DeepSeekModelForCausalLM
from data.data_loader import StreamingPretrainBatchDataset

# Mapping for torch autocast types
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def get_model(mode: str, model_config_path: Optional[str] = None, device: Optional[str] = None) -> DeepSeekModelForCausalLM:
    """
    Loads DeepSeek model from configuration and moves to specified device.

    Args:
        device: 'cpu' or 'cuda'
        mode: 'pretrain' or 'sft'

    Returns:
        Instantiated DeepSeekModelForCausalLM on device.
    """
    if model_config_path:
        model_config = DeepSeekConfig.from_json(model_config_path)
    else:
        config_path = {
            "sft": "config/sft.json",
            "pretrain": "config/pretrain.json",
        }.get(mode)
        if config_path is None:
            raise ValueError(f"Unknown mode: {mode}")
        model_config = DeepSeekConfig.from_json(config_path)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config.device = device

    model = DeepSeekModelForCausalLM(model_config).to(device)

    total_params, activated_params = model.get_total_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Activated parameters: {activated_params:,}")
    print(f"Activated ratio: {activated_params / total_params:.2%}")
    print(f"Model size (GB): {total_params * 2 / (1024**3):.2f} (assuming fp16)")
    return model



def configure_scheduler(optimizer, args) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Configures a learning rate scheduler.

    Supported types:
      - cosine decay
      - linear decay
      - one-cycle policy

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance.
        args (Namespace): Training arguments containing scheduler_type, decay settings.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: Scheduler object, or None if disabled.
    """
    if not args.decay_lr:
        return None

    if args.scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.lr_decay_iters, eta_min=args.min_learning_rate)
    elif args.scheduler_type == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args.min_learning_rate / args.learning_rate,
            total_iters=args.lr_decay_iters,
        )
    elif args.scheduler_type == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            total_steps=args.max_train_steps,
            pct_start=0.05,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {args.scheduler_type}")


def configure_optimizers(model, args) -> torch.optim.Optimizer:
    """
    Configures AdamW optimizer (or AdamW8bit if specified).

    Groups model parameters into two groups:
      - Parameters with weight decay (e.g., Linear/Conv weights)
      - Parameters without weight decay (e.g., biases, LayerNorm/Gamma)

    Rationale:
      - Applying weight decay to biases and LayerNorm scale parameters can hurt model quality.
      - See discussions in:
        - "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017) https://arxiv.org/abs/1711.05101
        - HuggingFace best practices (https://huggingface.co/docs/transformers/main/en/main_classes/optimizer_schedules#transformers.AdamW)

    Args:
        model (nn.Module): Model whose parameters are to be optimized.
        args (Namespace): Training arguments.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": args.adamw_weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    if args.use_eight_bit_optimizer:
        return bnb.optim.AdamW8bit(
            optim_groups,
            lr=args.learning_rate,
            betas=(args.adamw_beta1, args.adamw_beta2),
        )
    else:
        return torch.optim.AdamW(
            optim_groups,
            lr=args.learning_rate,
            betas=(args.adamw_beta1, args.adamw_beta2),
            fused=args.adamw_use_fused,
        )


def get_dataloader(args, split: str):
    """
    Instantiates a streaming dataset for training or validation.

    Args:
        args (Namespace): Contains dataset path, block size, batch size, etc.
        split (str): 'train' or 'validation'.

    Returns:
        StreamingPretrainBatchDataset: HuggingFace streaming loader wrapped into batches.
    """
    if not args.hf_dataset_name:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_tokenizer_name,
        model_max_length=args.pretrained_tokenizer_max_length,
        trust_remote_code=True,
    )

    return StreamingPretrainBatchDataset(
        pth=args.hf_dataset_name,
        name=args.hf_dataset_dir,
        tokenizer=tokenizer,
        block_size=args.seq_len,
        batch_size=args.batch_size,
        split=split,
    )


def load_checkpoint(resume_path: str, device: str, weights_only: bool = True):
    """
    Loads a saved checkpoint containing model and optimizer state.

    Args:
        resume_path (str): Path to .pt checkpoint file.
        device (str): Device for loaded tensors.

    Returns:
        dict: Checkpoint dictionary.
    """
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint file {resume_path} not found.")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=weights_only)
    return checkpoint

def safe_next(loader_iter, loader):
    try:
        batch = next(loader_iter)
        did_wrap = False
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
        did_wrap = True
    return batch, loader_iter, did_wrap

def train_loop(training_args: TrainingArgs, mode: str):
    """
    Main loop for either pretraining or supervised fine-tuning.

    Steps:
      1. Load model, optimizer, scheduler
      2. Stream batches
      3. Backpropagate loss (mixed precision if GPU)
      4. Save checkpoints
      5. Log metrics to WandB
      6. Periodic evaluation (optional)

    Args:
        training_args (Namespace): Training configuration from argparse.
        mode (str): 'pretrain' or 'sft'
    """
    ctx = nullcontext() if training_args.device == "cpu" else torch.amp.autocast(device_type=training_args.device, dtype=ptdtype[training_args.dtype])
    if training_args.device == 'cuda':
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    if training_args.wandb_log:
        wandb.init(project=training_args.wandb_project, name=training_args.wandb_run_name, config=vars(training_args))

    model = get_model(mode, training_args.model_config_path, training_args.device)
    optimizer = configure_optimizers(model, training_args)
    scheduler = configure_scheduler(optimizer, training_args)
    optimizer.zero_grad(set_to_none=True)

    train_loader = get_dataloader(training_args, split="train")
    val_loader = get_dataloader(training_args, split="validation") if mode == "sft" else None

    train_iter = iter(train_loader)
    val_iter = iter(val_loader) if val_loader else None

    epoch_num = 0
    iter_num = 0

    generation_tokenizer = AutoTokenizer.from_pretrained(
        training_args.pretrained_tokenizer_name,
        model_max_length=training_args.pretrained_tokenizer_max_length,
        trust_remote_code=True,
    )

    if training_args.resume:
        checkpoint = load_checkpoint(training_args.resume, training_args.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iter_num = checkpoint.get("iter_num", 0)

    while iter_num < training_args.max_train_steps:
        past_key_val = None
        optimizer.zero_grad(set_to_none=True)

        for i in range(training_args.gradient_accumulation_steps):
            batch, train_iter, epoch_increment = safe_next(train_iter, train_loader)
            epoch_num += epoch_increment
            x, y = batch["input_ids"].to(training_args.device), batch["labels"].to(training_args.device)
            with ctx:
                _, loss, past_key_val = model(x, y, past_key_value=past_key_val)
            if scaler is not None:
                scaler.scale(loss / training_args.gradient_accumulation_steps).backward()
            else:
                (loss / training_args.gradient_accumulation_steps).backward()

            # Detach past key values from the gradient graph (if we didn't do this, calling .backward() would result in a reference
            # to previous computation graphs).
            if past_key_val is not None:
                for layer_idx in range(len(past_key_val.key_cache)):
                    if past_key_val.key_cache[layer_idx] is not None:
                        past_key_val.key_cache[layer_idx] = past_key_val.key_cache[layer_idx].detach()
                    if past_key_val.value_cache[layer_idx] is not None:
                        past_key_val.value_cache[layer_idx] = past_key_val.value_cache[layer_idx].detach()
        
        if training_args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if (iter_num + 1) % training_args.log_interval == 0 and training_args.wandb_log:
            wandb.log({
                "Step": iter_num,
                "Epoch": epoch_num,
                "Train Loss": loss.item(),
                "Learning Rate": optimizer.param_groups[0]["lr"],
            })

        if (iter_num + 1) % training_args.save_interval == 0:
            save_path = os.path.join(training_args.out_dir, f"{training_args.checkpoint_path}_step{iter_num + 1}.pt")
            os.makedirs(training_args.out_dir, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model.config,
                "iter_num": iter_num + 1,
                "training_args": vars(training_args),
            }, save_path)

        if (iter_num + 1) % training_args.eval_interval == 0 and val_loader is not None:
            model.eval()
            val_losses = []
            for _ in range(training_args.eval_iters):
                batch, val_iter, _ = safe_next(val_iter, val_loader)
                x, y = batch["input_ids"].to(training_args.device), batch["labels"].to(training_args.device)
                _, val_loss_tensor, _ = model(x, y)
                val_losses.append(val_loss_tensor.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            if training_args.wandb_log:
                wandb.log({
                    "Validation Loss": avg_val_loss,
                    "Epoch": epoch_num,
                })
            model.train()
        
        if training_args.generate_interval > 0 and (iter_num + 1) % training_args.generate_interval == 0:
            input_ids = generation_tokenizer("the meaning of life is", return_tensors="pt").input_ids.to(training_args.device)
            with torch.no_grad():
                output_ids = model.generate(input_ids, max_length=50)
            decoded = generation_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print("\n[Sample Generation @ step", iter_num + 1, "]")
            print(decoded)
            print("=" * 80)
        iter_num += 1
