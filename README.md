# llm
## Minimum Working Example
In addition to unit + integration testing on mock models, we also provide 
a `pretrain` command which demonstrates training medium sized model with 500M parameters. After running the [setup](#setup) commands, you're ready to start training a language model!
```
uv run python3 train/pretrain.py  \
  --hf-dataset-name='roneneldan/TinyStories' \
  --hf-dataset-dir=default \
  --batch-size=8 \
  --seq-len=512 \
  --learning-rate=2e-4 \
  --out-dir=checkpoints/pretrain_medium \
  --wandb-run-name="pretrain_medium" \
  --model-config-path=config/medium_pretrain.json \
  --max-train-steps=10000 \
  --save-interval=1000 \
  --log-interval=50 \
  --grad-clip=1 \
  --gradient-accumulation-steps=1 \
  --generate-interval=1000 \
  --dtype=bfloat16
```

More interesting training experiments would involve choosing a larger dataset via argument `hf-dataset-[name|dir]` as well as specifying a larger model architecture
via the config file pointed to by `model-config-path`.

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

### Data Abstraction
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
