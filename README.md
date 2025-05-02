# A Miniature Language Model in <5k lines of code
> "Modern, modular language model pretraining that is simple enough for a single-GPU researcher, but flexible enough to be a launchpad for custom architectures."

## Comparison with NanoGPT + OLMoE + Llama
[NanoGPT](https://github.com/karpathy/nanoGPT) hasn't been updated significantly in
the past several years; during this elapsed time, MoE, RoPE, LoRA, and KV-Caching
have all become part of modern-stack language models.
[OLMo](https://github.com/allenai/OLMo) is an open source language model which has many
overlapping features, however, it does [not](https://github.com/allenai/OLMo/pull/639) have support for MoE. As an entirely separate project, [OLMoE](https://arxiv.org/abs/2409.02060) adds support for mixture of experts, however, the [instructions for pretraining](https://github.com/allenai/OLMoE/tree/main?tab=readme-ov-file#pretraining) are more of a manual recipe than an engineering framework. Perhaps [Llama-models](https://github.com/meta-llama/llama-models) is the closest to our featureset, with the notable difference that Llama is supported by a team of engineers, vs. our Miniature Language Model was created largely in a weekend by 1 person. 

## Minimum Working Example on Real Data
In addition to unit + integration testing on mock models, we also provide 
a `pretrain` command which demonstrates training medium sized model with 35M parameters. After running the [setup](#setup) commands, you're ready to start training a language model! An example of a small pre-training run can be found [here](https://wandb.ai/asantucci-stanford-university/tiny-deepseek-test/runs/4ktk4cl1).
```
uv run python3 minimal-example.py
```

More interesting training experiments would involve choosing a larger dataset via argument `hf-dataset-[name|dir]` within the `data` dictionary for `configs/train/tiny.yaml` as well as specifying a larger model architecture within the `model` dictionary for `configs/model/tiny.yaml`. A more purpose built `pretrain.py` and `sft.py` can be found within the `train/` subdirectory.

## Why Build a Language Module with Exposed Internals?
Building a training module with fully exposed internals offers a rare and valuable opportunity to deeply understand and optimize
modern language model architectures. Instead of relying on opaque APIs, this project surfaces the key design choices, such as:

  - attention layouts,
  - positional embeddings (ROPE, NOPE),
  - LoRA parameterizations, and
  - cache management

These govern the performance, efficiency, and scaling behavior of large models. By reimplementing these mechanisms from first principles while retaining modularity and flexibility, we make it easier to

  - debug training dynamics,
  - experiment with new methods (e.g., dynamic context extension, expert routing strategies), and
  - adapt core components for emerging tasks. 

This level of control is crucial for both researchers seeking conceptual clarity and engineers aiming to push the boundaries of efficient, large-scale model training.

## Non-Goals of This Project
### Distributed Computation
This project intentionally focuses on simplicity and local experimentation rather than scaling to distributed multi-node training.
We do not plan to add support for cluster orchestration systems such as Slurm, Kubernetes, or Ray. Instead, the goal is to maintain a
clean, self-contained codebase that runs efficiently on a single GPU or a single workstation. By constraining the scope, we prioritize
rapid iteration, easy customization, and deeper understanding of the underlying algorithms—without introducing the complexity
and fragility often associated with large distributed systems.

### Tokenizer Agnostic
We do not implement a Tokenizer from scratch, but instead prefer to keep this as an abstraction. For more explanation, see: [data/README.md](data/README.md)

### Data Scraping
We intentionally avoid scraping, curating, or maintaining large-scale internet datasets within this repository.  
Instead, this project assumes that a dataset can be accessed via a standard Hugging Face dataset loader.

This decision is deliberate:
- **Simplified Scope**: Scraping, cleaning, and tokenizing internet-scale corpora is a complex, multi-month effort involving privacy, legality, and ethical considerations outside the core goal of model development.
- **Portability**: By treating datasets as a pluggable interface, users can easily swap in their own preprocessed datasets without modifying model internals.
- **Focus on Model Research**: We prioritize experiments on model architectures, optimization techniques, and training dynamics—not data engineering.

For prototyping, we provide examples of using existing Hugging Face datasets that require no scraping:
- [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) (synthetic language corpus)
- [`wikitext`](https://huggingface.co/datasets/wikitext) (Wikipedia-derived dataset)
- [`openwebtext`](https://huggingface.co/datasets/openwebtext) (web text reproduction)

In practice, **production-grade LLMs** require specialized data pipelines—but building those is outside the scope of this repository.

## Setup
After cloning the repository, set up a UV environment via `uv init` within the repo directory. Then, install dependencies using `uv pip install -e .`.

## Testing
To run all tests, execute using `uv run pytest`.
