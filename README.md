# DynamicsBench — private usability candidate

Validate deterministic PyTorch simulation states and parameter gradients, rank eligible Euler/RK4 configurations by measured execution cost, and export a scoped recommendation.

**Start with [TESTING.md](TESTING.md).** This is the frozen 0.1.0rc1 P3 candidate. External human usability is pending. [CANDIDATE.json](CANDIDATE.json) records exact file hashes; the Git tag identifies the full tester packet.

Python >=3.11 is required. CPU is the default test environment; NVIDIA hardware and PACE access are not required. Source, an installable wheel, a minimal analytical example and an independent MIT-licensed neural-ODE example are included.

See [the product guide](docs/PRODUCT_GUIDE.md) for the API, CLI and export contracts. A caller-supplied reference must be independently justified. Passing sampled numerical checks does not guarantee prediction quality, training convergence or future-checkpoint accuracy. Held-out cases do not influence selection, and overlapping timing ranges make ranks uncertain.

This repository is a small tester distribution. The complete research archive and machine-specific setup stay outside it. The research study and compilation measurements are separate from this usability exercise; see [EVIDENCE.md](EVIDENCE.md). Licensing status and third-party notices are in [LICENSE_STATUS.md](LICENSE_STATUS.md).
