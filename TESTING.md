# Independent usability test

Audience: comfortable with Python/PyTorch, unfamiliar with DynamicsBench. Use your own local machine and a fresh environment. CPU is preferred. If local setup is unsuitable, record that and use your own cluster account with a fresh environment; do not reuse the developer's prepared installation.

## Setup

Clone/download this private repository, then select the tag `p3-v0.1.0rc1` and work from its root. Record your OS, Python version and starting time. Verify your Python is at least3.11 (`python3 --version`); choose another interpreter if necessary.

```sh
python3 -m venv .venv
```

macOS/Linux:
```sh
.venv/bin/python -m pip install "dist/dynamicsbench-0.1.0rc1-py3-none-any.whl[independent]"
.venv/bin/dynamicsbench --workload examples/product_independent.py --output results/tester-first-run
```

Windows PowerShell:
```powershell
.venv\Scripts\python.exe -m pip install "dist/dynamicsbench-0.1.0rc1-py3-none-any.whl[independent]"
.venv\Scripts\dynamicsbench.exe --workload examples/product_independent.py --output results/tester-first-run
```

Windows has not yet been independently verified; report any failure. Internet access is needed to install dependencies. GPU is optional and can be tested afterward on a documented compatible profile. Retain the terminal output and exact commands you used.

## Task A — first use and interpretation

Produce a complete report with the supplied independent workload. Locate the selected configuration, gradient/state evidence, eligible rankings and held-out outcome. Explain in your own words what the recommendation establishes, what it does not establish, and how you would react to heldout_failed or no_recommendation. Consult the README/product guide as needed; record what you consulted and where you needed help. Do not edit package internals.

## Task B — integrate your existing workload

If available, use a pre-existing deterministic dynamics function AND scalar objective of your own. Before scheduling, tell the organizer its device/dtype, state shape, named parameters and reference approach; this screens feasibility, not step-by-step implementation. Use examples/product_linear.py as the factory template and the guide as the contract. Supply an analytical or otherwise independently justified differentiable reference, candidates, tolerances and at least one snapshot. Run the CLI and retain its outputs. A failed candidate is valid evidence. Do not invent an entire new model/reference solely for this session.

If no suitable workload is available, record Task B as not attempted; Task A still tests onboarding, but does not establish independent custom-workload reuse. A supplied example with changed parameters is not equivalent to bringing your own model/objective.

## Feedback and evidence

Copy FEEDBACK_TEMPLATE.md and fill it in. Share the completed form, terminal log and the output directories (report/recommendation/provenance/rankings/lock). For custom work, include the adapter and source only if you can share them; remove private data and describe anything withheld. Check exported snapshots and absolute source paths before sharing. Record every intervention, including who helped and when. Assistance is allowed and must remain visible.

Suggested target is30–45minutes after dependency installation; it is not a pass deadline. Stop and report if blocked. The developer will retain the attempt, fix demonstrated friction and ask for a focused retest against a separately identified revision. Do not overwrite first-attempt artifacts.
