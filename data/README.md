# Data Utilities

This directory contains lightweight helpers for batching and preparing inputs for pretraining and supervised fine-tuning (SFT).

## Why Not Roll Our Own Tokenizer?

While exposing internals for model components (e.g., self-attention, Mixture-of-Experts routing) is a core design philosophy of this project, we deliberately choose **not** to reimplement a tokenizer for the following reasons:

- **Tokenizer Engineering is Orthogonal**:  
  Building a tokenizer (e.g., BPE, WordPiece, Unigram LM) is an interesting but separate field from studying transformer model architecture itself.

- **Efficiency and Correctness**:  
  Libraries like Hugging Face's `AutoTokenizer` are heavily optimized for speed, memory, and correctness across hundreds of tokenizer types and datasets.

- **Focus on Core Modeling**:  
  Our goal is to study and expose the inner workings of modern pretraining, fine-tuning, and inference systems — tokenization is treated as a solved utility.

- **Ease of Integration**:  
  Hugging Face tokenizers integrate seamlessly with HF datasets, streaming, and distributed data loading without additional engineering overhead.

If needed in the future, a custom tokenizer could easily be integrated without affecting the design of the rest of the training or evaluation pipelines.

## Included Utilities

| Module  | Purpose |
|:--------|:--------|
| `pad.py`  | Manual padding of sequences to the maximum batch length, using a specified padding value. |
| `collators.py` | Formatting, padding, and masking examples for SFT and pretraining modes. |
