# DynamicsBench

Validate differentiable PyTorch simulations before choosing a cheaper solver.

DynamicsBench compares observation trajectories and parameter gradients against
a caller-supplied reference, measures complete forward/backward execution cost,
and exports a recommendation scoped to the cases you tested.

**v0.1 is an experimental engineering tool with a focused empirical study.**
It is intended for researchers and engineers with deterministic dynamics,
explicit tensor parameters, a scalar objective, and a defensible reference.
CPU works out of the box; a GPU or cluster account is not required.

## Try it

Python 3.11 or newer is required. From a terminal with Git and Python:

```sh
git clone https://github.com/yixuan-zheng/DynamicsBench.git
cd DynamicsBench
python3 -m venv .venv
.venv/bin/python -m pip install "."
.venv/bin/dynamicsbench --workload examples/product_linear.py --output results/first-run
```

Windows and user-space installation commands are in [QUICKSTART.md](QUICKSTART.md).
The same guide covers the release wheel, optional upstream neural-ODE example,
and a minimal Python API call. This release is distributed on GitHub, not PyPI.

A successful run exports:

| File | Purpose |
|---|---|
| report.json | State/gradient checks, failures, timings and scope |
| rankings.csv | Passing configurations ranked by median cost |
| recommendation.json | Selected solver and held-out result |
| provenance.json | Source hashes and environment provenance |
| environment-lock.txt | Installed distribution versions |

An illustrative excerpt from the analytical example:

```json
{"status": "selected", "configuration": {"method": "rk4", "steps": 4}}
```

The exact JSON contract is in the [product guide](docs/PRODUCT_GUIDE.md).
Timing ranks can differ across machines and overlapping timing ranges are reported.
Create a new output directory for each run.

## Bring your workload

Define `build_workload(device="cpu")` returning `Workload`, then pass its Python file
to the CLI. Supply dynamics, objective, differentiable reference, named parameter
snapshots, candidate Euler/RK4 settings, and optional held-out cases.
[examples/product_linear.py](examples/product_linear.py) is the smallest complete
example. The [API guide](docs/PRODUCT_GUIDE.md) documents callback contracts and
the `selected`, `no_recommendation`, and `heldout_failed` outcomes.

## What the study found

A frozen nonlinear trajectory-learning study used two difficulty regimes and
three paired seeds per regime (18 training runs). Initialization-only selection
saved **18.35–22.01% charged policy time** in the easier regime with similar
held-out quality. In the harder regime, finer selected solvers improved sampled
numerical fidelity but cost **2.14–2.31×** the fixed baseline without improving
prediction quality. The added early validation checkpoint provided no incremental
useful decision under the frozen criterion.

These bounded findings distinguish numerical agreement from useful training
decisions. A separate analytical V100 compilation experiment observed 4.72–7.45×
steady-state speedups, with substantial startup cost; it is not a training speedup.
See the [technical report](TECHNICAL_REPORT.md), [evidence index](EVIDENCE_INDEX.md),
and [reproduction guide](REPRODUCING.md).

## Scope and limitations

The product CLI uses eager Euler/RK4 on equally spaced observations with
deterministic CPU/CUDA float32/float64 snapshots. It does not automatically justify
the reference or certify physical validity, future checkpoints, convergence,
stiff systems, stochastic/stateful models, energy use, or downstream quality.
Validation and reference construction can cost more than they save.

One external Windows tester completed the supplied example and a targeted retest.
Integration of an independent user's own model remains untested. Compatibility
claims apply to the [recorded profiles](docs/COMPATIBILITY.md), not every dependency
combination. Historical research replay uses a separate environment and source tree.

## Feedback and license

Open an issue with your framework, differentiation method, recent solver-validation
problem and a small reproducible example. Review exports before sharing: caller
metadata, exception messages and model snapshots can contain sensitive information.
See [CONTRIBUTING.md](CONTRIBUTING.md).

MIT licensed. Upstream attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Developed by Yixuan Zheng, a master's student in Computer Science at Georgia Tech.
