# Feedback and contributions

For a bug, provide the release/commit, Python/PyTorch versions, CPU/GPU information,
command, traceback and smallest workload reproducing the issue. Review all shared
artifacts for private paths, data and parameters. Do not post credentials.

For a workflow discussion, describe a recent numerical decision, your current
validation process, its cost and what a useful output would change. No installation
or proprietary code sharing is required to discuss the problem.

Install `.[independent]` in a virtual environment and run
`python -m unittest discover -s tests` before proposing code changes. CUDA tests
skip when CUDA is unavailable. Scientific replay is separate from the product.
Contributions are supplied under this repository's MIT license.
