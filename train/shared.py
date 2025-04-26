import json
import os
import torch
import wandb
from contextlib import nullcontext
from typing import Optional

import bitsandbytes as bnb
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR

from config.deepseek import DeepSeekConfig
from model.deepseek import DeepSeekModelForCausalLM
from data.data_loader import StreamingPretrainBatchDataset

# Mapping for torch autocast types
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def get_model(device: str, mode: str) -> DeepSeekModelForCausalLM:
    """
    Loads DeepSeek model from configuration and moves to specified device.

    Args:
        device: 'cpu' or 'cuda'
        mode: 'pretrain' or 'sft'

    Returns:
        Instantiated DeepSeekModelForCausalLM on device.
    """
    config_path = {
        "sft": "config/sft.json",
        "pretrain": "config/pretrain.json",
    }.get(mode)

    if config_path is None:
        raise ValueError(f"Unknown mode: {mode}")

    config = DeepSeekConfig.from_json(config_path)
    model = DeepSeekModelForCausalLM(config)
    model.to(device)

    total_params, activated_params = model.get_total_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Activated parameters: {activated_params:,}")
    print(f"Activated ratio: {activated_params / total_params:.2%}")
    return model


def configure_scheduler(optimizer, args) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Configures learning rate scheduler based on args.

    Args:
        optimizer: Torch optimizer instance
        args: Training arguments namespace

    Returns:
        Scheduler object or None if disabled
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
    Groups model parameters and configures optimizer.

    Args:
        model: Model with parameters
        args: Training arguments

    Returns:
        Configured optimizer.
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
    Creates a streaming dataset loader using HuggingFace datasets.

    Args:
        args: Training arguments
        split: 'train' or 'validation'

    Returns:
        StreamingPretrainBatchDataset
    """
    if not args.hf_dataset_name:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-6.7b-base",
        model_max_length=1_000_000,
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


def load_checkpoint(resume_path: str, device: str):
    """
    Loads model and optimizer states from checkpoint.

    Args:
        resume_path: Path to checkpoint .pt file
        device: Target device

    Returns:
        checkpoint dict
    """
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint file {resume_path} not found.")
    checkpoint = torch.load(resume_path, map_location=device)
    return checkpoint


def train_loop(args, mode: str):
    """
    Main supervised fine-tuning or pretraining loop.

    Args:
        args: Training arguments
        mode: 'pretrain' or 'sft'
    """
    ctx = nullcontext() if args.device == "cpu" else torch.amp.autocast(device_type=args.device, dtype=ptdtype[args.dtype])

    if args.wandb_log:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    model = get_model(args.device, mode)
    optimizer = configure_optimizers(model, args)
    scheduler = configure_scheduler(optimizer, args)
    optimizer.zero_grad(set_to_none=True)

    train_loader = get_dataloader(args, split="train")
    val_loader = get_dataloader(args, split="validation") if mode == "sft" else None

    train_iter = iter(train_loader)
    val_iter = iter(val_loader) if val_loader else None

    iter_num = 0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, args.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iter_num = checkpoint.get("iter_num", 0)

    while iter_num < args.max_train_steps:
        past_key_val = None
        optimizer.zero_grad(set_to_none=True)

        for _ in range(args.gradient_accumulation_steps):
            batch = next(train_iter)
            x, y = batch["input_ids"].to(args.device), batch["labels"].to(args.device)

            with ctx:
                _, loss, past_key_val = model(x, y, past_key_value=past_key_val)

            (loss / args.gradient_accumulation_steps).backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (iter_num + 1) % args.log_interval == 0 and args.wandb_log:
            wandb.log({
                "Step": iter_num,
                "Train Loss": loss.item(),
                "Learning Rate": optimizer.param_groups[0]["lr"],
            })

        if (iter_num + 1) % args.save_interval == 0:
            save_path = os.path.join(args.out_dir, f"{args.checkpoint_path}_step{iter_num + 1}.pt")
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model.config,
                "iter_num": iter_num + 1,
                "training_args": vars(args),
            }, save_path)

        if (iter_num + 1) % args.eval_interval == 0 and val_loader is not None:
            model.eval()
            val_losses = []
            for _ in range(args.eval_iters):
                batch = next(val_iter)
                x, y = batch["input_ids"].to(args.device), batch["labels"].to(args.device)
                _, val_loss_tensor, _ = model(x, y)
                val_losses.append(val_loss_tensor.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            wandb.log({"Validation Loss": avg_val_loss})
            model.train()

        iter_num += 1
