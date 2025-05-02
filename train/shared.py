"""
Training utilities for DeepSeek-style models.

Design:
- Single-GPU friendly
- Mixed-precision supported (via torch.amp.autocast)
- Lightweight (no distributed training, no DeepSpeed/FSDP)
"""
from collections import deque
import numpy as np
from dataclasses import dataclass
import gc
import os
import torch
import wandb
from contextlib import nullcontext
import time
from typing import Optional
from omegaconf import OmegaConf
from scipy.stats import linregress

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

def get_model(mode: str, model_cfg: dict, device: Optional[str] = None) -> DeepSeekModelForCausalLM:
    """
    Loads DeepSeek model from configuration dictionary and moves to specified device.

    Args:
        model_cfg: dictionary or OmegaConf loaded model configuration.
        mode: 'pretrain' or 'sft'
        device: 'cpu' or 'cuda'. If None, auto-detect.

    Returns:
        Instantiated DeepSeekModelForCausalLM on the specified device.
    """
    # Handle device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Keeps the relevant fields from the passed configuration dict.
    model_fields = {k: model_cfg[k] for k in DeepSeekConfig.__dataclass_fields__.keys() if k in model_cfg}
    # Build DeepSeekConfig from model_cfg dict
    model_config = DeepSeekConfig(**model_fields)
    model_config.device = device

    # Instantiate model
    model = DeepSeekModelForCausalLM(model_config).to(device)

    # Print model stats
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

class TrainingLossMonitor:
    """
    Monitors recent training losses to detect divergence or stagnation in pretraining.

    Early-stops training if:
      - Loss rises consistently across several logging intervals
      - Loss stays flat within a small delta across several intervals

    Attributes:
        patience (int): Number of logging intervals to wait before checking early stop.
        relative_flat_tolerance (float): Relative change to consider loss "flat." Defaults to 0.5%.
        min_steps_before_check (int): Minimum number of total steps before checking early stopping.
    """

    def __init__(self, patience=4, relative_flat_tolerance=0.005, min_steps_before_check=1000):
        self.patience = patience
        self.relative_flat_tolerance = relative_flat_tolerance
        self.min_steps_before_check = min_steps_before_check
        self.loss_history = deque(maxlen=patience)
        self.step_history = deque(maxlen=patience)

    def update(self, loss: float, step: int):
        """
        Updates the internal state with new training loss at a given step.

        Args:
            loss (float): Training loss at current step.
            step (int): Current training step.
        """
        self.loss_history.append(loss)
        self.step_history.append(step)

    def should_stop(self) -> bool:
        """
        Determines whether early stopping should be triggered based on recent losses.

        Returns:
            bool: True if training should be stopped early, False otherwise.
        """
        if len(self.loss_history) < self.patience:
            return False  # Not enough history yet

        if self.step_history[-1] < self.min_steps_before_check:
            return False  # Don't check before enough steps

        deltas = [self.loss_history[i] - self.loss_history[i-1] for i in range(1, len(self.loss_history))]
        num_rising = sum(d > 0 for d in deltas)
        relative_deltas = [
            abs((self.loss_history[i] - self.loss_history[i-1]) / max(self.loss_history[i-1], 1e-6))
            for i in range(1, len(self.loss_history))
        ]

        num_flat_pct = sum(delta < self.relative_flat_tolerance for delta in relative_deltas)

        if num_rising >= self.patience - 1 or num_flat_pct >= self.patience - 1:
            return True

        return False

class LinearSlopeBasedEarlyStopper:
    def __init__(self,
                 window_size=250,
                 slope_history_window=20,
                 min_steps_before_check=1000,
                 ema_beta=0.9,
                 slope_mean_threshold=0.003,
                 slope_magnitude_threshold=0.002,
                 slope_fraction_threshold=0.7,
                 wandb_logger=None):
        
        self.window_size = window_size
        self.slope_history_window = slope_history_window
        self.min_steps_before_check = min_steps_before_check
        self.ema_beta = ema_beta

        self.slope_mean_threshold = slope_mean_threshold
        self.slope_magnitude_threshold = slope_magnitude_threshold
        self.slope_fraction_threshold = slope_fraction_threshold
        self.wandb_logger = wandb_logger

        self.loss_history = deque(maxlen=window_size)
        self.step_history = deque(maxlen=window_size)
        self.slope_history = deque(maxlen=slope_history_window)

        self.ema_loss = None

    def update(self, loss: float, step: int):
        # Smooth loss with EMA
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.ema_beta * self.ema_loss + (1 - self.ema_beta) * loss

        self.loss_history.append(self.ema_loss)
        self.step_history.append(step)

        if len(self.loss_history) == self.window_size:
            x = np.array(self.step_history)
            y = np.array(self.loss_history)
            slope, *_ = linregress(x, y)
            self.slope_history.append(slope)

            if self.wandb_logger:
                self.wandb_logger.log({
                    "EarlyStopper/Slope": slope,
                    "EarlyStopper/SlopeMean": np.mean(self.slope_history) if self.slope_history else 0,
                    "EarlyStopper/FractionAboveMagnitude": np.mean([s > self.slope_magnitude_threshold for s in self.slope_history]) if self.slope_history else 0,
                    "EarlyStopper/Step": step,
                })

    def should_stop(self) -> bool:
        if len(self.loss_history) < self.window_size:
            return False
        if self.step_history[-1] < self.min_steps_before_check:
            return False
        if len(self.slope_history) < self.slope_history_window:
            return False

        slope_mean = np.mean(self.slope_history)
        slope_frac_exceeds = np.mean([s > self.slope_magnitude_threshold for s in self.slope_history])

        if slope_mean > self.slope_mean_threshold and slope_frac_exceeds > self.slope_fraction_threshold:
            print(f"[EarlyStop] Triggered at step {self.step_history[-1]} | "
                  f"slope_mean={slope_mean:.5f}, "
                  f"frac_exceeds={slope_frac_exceeds:.2f}")
            return True
        return False

def train_loop(cfg: OmegaConf, mode: str, loss_monitor: Optional[TrainingLossMonitor | LinearSlopeBasedEarlyStopper] = None, wandb_logger = None):
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
        loss_monitor (Optional[TrainingLossMonitor]): Monitor for early stopping, if provided.
    """
    ctx = nullcontext() if cfg.device == "cpu" else torch.amp.autocast(device_type=cfg.device, dtype=ptdtype[cfg.dtype])
    if cfg.device == 'cuda':
        scaler = torch.amp.GradScaler()  # <-- This is the constructor that depends on GPU config.
    else:
        scaler = None

    model = get_model(mode, cfg, cfg.device)
    optimizer = configure_optimizers(model, cfg)
    scheduler = configure_scheduler(optimizer, cfg)
    optimizer.zero_grad(set_to_none=True)

    train_loader = get_dataloader(cfg, split="train")
    val_loader = get_dataloader(cfg, split="validation") if mode == "sft" else None

    train_iter = iter(train_loader)
    val_iter = iter(val_loader) if val_loader else None

    epoch_num = 0
    iter_num = 0

    generation_tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_tokenizer_name,
        model_max_length=cfg.pretrained_tokenizer_max_length,
        trust_remote_code=True,
    )

    if cfg.resume:
        checkpoint = load_checkpoint(cfg.resume, cfg.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iter_num = checkpoint.get("iter_num", 0)

    while iter_num < cfg.max_train_steps:
        past_key_val = None
        optimizer.zero_grad(set_to_none=True)

        for i in range(cfg.gradient_accumulation_steps):
            batch, epoch_increment = train_loader.safe_next()
            epoch_num += epoch_increment
            x, y = batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device)
            with ctx:
                _, loss, past_key_val = model(x, y, past_key_value=past_key_val)
            if scaler is not None:
                scaler.scale(loss / cfg.gradient_accumulation_steps).backward()
            else:
                (loss / cfg.gradient_accumulation_steps).backward()

            # Detach past key values from the gradient graph (if we didn't do this, calling .backward() would result in a reference
            # to previous computation graphs).
            if past_key_val is not None:
                for layer_idx in range(len(past_key_val.key_cache)):
                    if past_key_val.key_cache[layer_idx] is not None:
                        past_key_val.key_cache[layer_idx] = past_key_val.key_cache[layer_idx].detach()
                    if past_key_val.value_cache[layer_idx] is not None:
                        past_key_val.value_cache[layer_idx] = past_key_val.value_cache[layer_idx].detach()
        
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if (iter_num + 1) % cfg.log_interval == 0 and cfg.wandb_log and wandb_logger is not None:
            wandb_logger.log({
                "Step": iter_num,
                "Epoch": epoch_num,
                "Train Loss": loss.item(),
                "Learning Rate": optimizer.param_groups[0]["lr"],
            })
            if loss_monitor is not None:
                loss_monitor.update(loss.item(), iter_num + 1)

                if loss_monitor.should_stop():
                    print(f"Early stopping sweep run at step {iter_num+1} due to bad training dynamics.")
                    break

        if (iter_num + 1) % cfg.save_interval == 0:
            save_path = os.path.join(cfg.out_dir, f"{cfg.checkpoint_path}_step{iter_num + 1}.pt")
            os.makedirs(cfg.out_dir, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model.config,
                "iter_num": iter_num + 1,
                "training_args": vars(cfg),
            }, save_path)

        if (iter_num + 1) % cfg.eval_interval == 0 and val_loader is not None:
            model.eval()
            val_losses = []
            for _ in range(cfg.eval_iters):
                batch, _ = val_loader.safe_next()
                x, y = batch["input_ids"].to(cfg.device), batch["labels"].to(cfg.device)
                _, val_loss_tensor, _ = model(x, y)
                val_losses.append(val_loss_tensor.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            if cfg.wandb_log and wandb_logger is not None:
                wandb_logger.log({
                    "Validation Loss": avg_val_loss,
                    "Epoch": epoch_num,
                })
            model.train()
        
        #if cfg.generate_interval > 0 and (iter_num + 1) % cfg.generate_interval == 0:
        #    input_ids = generation_tokenizer("the meaning of life is", return_tensors="pt").input_ids.to(cfg.device)
        #    with torch.no_grad():
        #        output_ids = model.generate(input_ids, max_length=50)
        #    decoded = generation_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        #    print("\n[Sample Generation @ step", iter_num + 1, "]")
        #    print(decoded)
        #    print("=" * 80)
        iter_num += 1
