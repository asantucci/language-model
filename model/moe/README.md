# Mixture-of-Experts (MoE) Layer
✅ Single-GPU Friendly  
✅ Modular and Extensible  
✅ Designed for Clarity and Experimentation

## Overview

This module implements a **Mixture-of-Experts** (MoE) layer inspired by architectures such as DeepSeek-V2, Switch Transformer, and GShard.

It features:
- Sparse expert routing using top-k softmax gating.
- Efficient batching and distribution of tokens across experts (`Distributor` class).
- Auxiliary loss injection for load balancing.
- Shared experts for fallback signal stability.
- Optimized batched matrix multiplications for GPU efficiency.

The code prioritizes **transparency, modularity, and pedagogical clarity**, making it highly suitable for **researchers, students, and hobbyists**.

## Key Features

- **Top-k routing:**  
  Each token is routed to only a small number of experts, reducing computational cost.

- **Efficient expert batching:**  
  Tokens routed to the same expert are grouped together for efficient processing.

- **Auxiliary load balancing loss:**  
  Encourages uniform expert utilization to prevent collapse onto a small subset of experts.

- **Shared experts:**  
  Provide universal fallback features for tokens that may not be adequately routed.

- **Clear and modular design:**  
  Each responsibility is encapsulated in its own class, with detailed inline documentation.

## Pros

✅ **Research-aligned MoE architecture**  
✅ **Highly readable and modifiable**  
✅ **Optimized token grouping for matrix multiplications**  
✅ **Correct gradient handling via custom autograd (`AddAuxiliaryLoss`)**  
✅ **Full unit test coverage on critical behaviors**  

## Limitations

⚡ **Memory scales linearly with number of tokens and experts.**  
⚡ **No capacity limit enforcement per expert (tokens may pile onto a few experts).**  
⚡ **All experts live on the same device (single-GPU design).**  
⚡ **No dynamic overflow handling or expert capacity redistribution.**  
⚡ **Simple top-k gating without noise or dropout regularization.**

**These are intentional design choices to favor simplicity and single-GPU friendliness.**

## Optional Upgrades

If you wish to extend the system for more advanced use cases:

- **Expert Capacity Limits:**  
  Prevent experts from being overloaded by enforcing a maximum number of routed tokens.

- **Routing Noise Injection:**  
  Add stochastic noise (e.g., Gumbel noise) during routing to improve generalization.

- **Expert Dropout:**  
  Randomly drop experts at training time for regularization.

- **Distributed Expert Sharding:**  
  Partition experts across multiple GPUs or nodes for scalability.

- **Dynamic Load Balancing:**  
  Adaptively bias tokens towards underloaded experts during routing.

## Pedagogical Value

Isolating the `Distributor` class and MoE internals provides valuable insight into:
- Sparse routing strategies.
- Efficient GPU batching for variable group sizes.
- Trade-offs between routing precision and system scalability.
- How auxiliary loss terms can be injected cleanly into the training objective.

The modularity makes this codebase an **excellent starting point for learning** and for
**building experimental large-scale mixture models** without needing massive compute resources.
