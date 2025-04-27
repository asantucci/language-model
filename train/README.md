## Status: Stable | Scope: Single-GPU Pretraining and Fine-Tuning
# `train/` Module

This directory houses lightweight training utilities for DeepSeek-style pretraining and supervised fine-tuning (SFT).

## Pretraining and Fine-Tuning Modes

The shared `train_loop(args, mode)` supports two modes:
- **Pretraining** (`mode="pretrain"`): unsupervised language model training
- **Supervised Fine-Tuning (SFT)** (`mode="sft"`): instruction tuning on labeled data

Both `pretrain.py` and `sft.py` configure arguments and call into this shared loop.

## Design Philosophy

- **Single-GPU, Single-Machine:**  
  No distributed training (`torch.distributed`, `DeepSpeed`, `FSDP`, `SLURM`, etc.).
- **Mixed-Precision Training:**  
  Transparent autocast with `torch.amp.autocast` when running on GPU.
- **Streaming Datasets:**  
  Data is streamed on-the-fly with HuggingFace `datasets` to minimize RAM usage.
- **Checkpoint Safety:**  
  Saved checkpoints include model weights, optimizer state, configs, and training step.

## Quick Usage

Pretraining:
```bash
python pretrain.py --hf-dataset-name "wikitext" --hf-dataset-dir "wikitext-2-raw-v1"
```

Supervised Fine-Tuning:
```bash
python sft.py --hf-dataset-name "openai_human_feedback" --hf-dataset-dir "rlhf-reward"
```

## Future Enhancements

- [ ] Early stopping support
- [ ] Validation metrics beyond loss (e.g., perplexity)
- [ ] Partial checkpoint loading (e.g., load only model weights)
- [ ] `Trainer` class abstraction (optional)

## Why Keep It Minimal?

This project deliberately favors clarity over heavy abstraction:
- No HuggingFace `Trainer`
- No DeepSpeed, Megatron-LM, or FSDP
- No multi-GPU orchestration

The aim is to **expose internals** while remaining **easy to audit, modify, and debug**.

If large-scale scaling is needed later, this codebase can be naturally extended.
