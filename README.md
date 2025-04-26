# llm
Minimum Working Example for language modeling.

## Why Build a Language Module with Exposed Internals?
Building a training module with fully exposed internals offers a rare and valuable opportunity to deeply understand and optimize
modern language model architectures. Instead of relying on opaque APIs, this project surfaces the key design choices—such as attention
layouts, positional embeddings (ROPE, NOPE), LoRA parameterizations, and cache management—which govern the performance, efficiency, and
scaling behavior of large models. By reimplementing these mechanisms from first principles while retaining modularity and flexibility,
we make it easier to debug training dynamics, experiment with new methods (e.g., dynamic context extension, expert routing strategies),
and adapt core components for emerging tasks. This level of control is crucial for both researchers seeking conceptual clarity and engineers
aiming to push the boundaries of efficient, large-scale model training.

## Non-Goals of This Project: Distributed Computation
This project intentionally focuses on simplicity and local experimentation rather than scaling to distributed multi-node training.
We do not plan to add support for cluster orchestration systems such as Slurm, Kubernetes, or Ray. Instead, the goal is to maintain a
clean, self-contained codebase that runs efficiently on a single GPU or a single workstation. By constraining the scope, we prioritize
rapid iteration, easy customization, and deeper understanding of the underlying algorithms—without introducing the complexity
and fragility often associated with large distributed systems.

## Setup
After cloning the repository, set up a UV environment via `uv init` within the repo directory. Then, install dependencies using `uv pip install -e .`.

## Testing
To run all tests, execute using `uv run pytest`.
