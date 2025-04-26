# DeepSeek Transformer

## Overview

This module implements a clean and modular DeepSeek-style Transformer architecture for language modeling (LM) and experimentation with MoE (Mixture-of-Experts).

Key features:
- Pre-Norm Multi-Head Self-Attention (MHSA) with rotary positional embeddings.
- Feedforward layers optionally replaced with a routed MoE layer.
- KVCache support for efficient autoregressive decoding.
- Lightweight design, easy to extend for both research and production.

---

## Structure

- `TransformerBlock`:
  - Pre-layernorm ➔ MHSA ➔ residual ➔ MoE/MLP ➔ residual.
- `DeepSeekTransformer`:
  - Stack of TransformerBlocks over token embeddings.
- `DeepSeekModelForCausalLM`:
  - Adds a linear LM head for language modeling tasks.

---

## Notable Details

- **Causal masking** and **rotary positional embeddings** are handled inside the attention layer.
- **KVCache** is used during generation to avoid recomputing keys/values across time.
- **Mixture-of-Experts (MoE)** routing is top-k sparse and load-balanced.
- **Initialization** follows DeepSeek: Normal(0, 0.006) standard deviation.

---

## References
- [DeepSeek LLM (arXiv:2401.02954)](https://arxiv.org/pdf/2401.02954)
- [DeepSeek MoE (arXiv:2401.06066)](https://arxiv.org/abs/2401.06066)
- [DeepSeek V2 Lite HuggingFace Repo](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
- [GPT-3 Paper (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [Switch Transformer (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961) for MoE routing

## Non-Goals

- We intentionally **avoid** cluster or Slurm job launchers.
- The design philosophy is **single GPU**, **small batch**, and **local training** to facilitate fast iteration.

## Potential Upgrades

- Enable FlashAttention kernels.
- Add curriculum sampling during pretraining.
- Experiment with higher top-k MoE routing (e.g., top-4).
- Expand support for weight quantization during generation.
