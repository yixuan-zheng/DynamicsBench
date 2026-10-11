# Independent usability test

Audience: comfortable with Python/PyTorch, unfamiliar with DynamicsBench. Use your own local machine and a fresh environment. CPU is preferred. If local setup is unsuitable, record that and use your own cluster account with a fresh environment; do not reuse the developer's prepared installation.

## Setup

Clone/download the repository at tag `p3-v0.1.0rc2`. Preserve your rc1 environment and outputs. Use a new environment and output directory. Record OS, Python version, commands and timing.

### Windows user-space route (uv)

If Python is missing or system installation is unavailable, obtain uv through its [official installer or standalone binaries](https://docs.astral.sh/uv/getting-started/installation/). Use an installation method permitted on your workstation. Restart the terminal if needed so `uv --version` works. Then, from the repository root:

```powershell
uv python install 3.11
uv venv .venv-rc2 --python 3.11
uv pip install --python .venv-rc2\Scripts\python.exe "dist/dynamicsbench-0.1.0rc2-py3-none-any.whl[independent]"
.\.venv-rc2\Scripts\dynamicsbench.exe --workload examples/product_independent.py --output results/tester-rc2
```

uv environments need not contain pip: `uv pip install` installs into the selected environment without invoking its pip module. Do not substitute `python -m pip` unless that environment actually has pip. See [uv environments](https://docs.astral.sh/uv/pip/environments/) and [managed Python](https://docs.astral.sh/uv/guides/install-python/).

### Existing Python route

Choose Python >=3.11. Standard Python venv normally bootstraps pip; if your manager omits it, use the uv route above or `python -m ensurepip --upgrade` where supported.

Windows PowerShell:
```powershell
py -3.11 -m venv .venv-rc2
.\.venv-rc2\Scripts\python.exe -m pip install "dist/dynamicsbench-0.1.0rc2-py3-none-any.whl[independent]"
.\.venv-rc2\Scripts\dynamicsbench.exe --workload examples/product_independent.py --output results/tester-rc2
```

macOS/Linux (choose a Python >=3.11 interpreter):
```sh
python3 -m venv .venv-rc2
.venv-rc2/bin/python -m pip install "dist/dynamicsbench-0.1.0rc2-py3-none-any.whl[independent]"
.venv-rc2/bin/dynamicsbench --workload examples/product_independent.py --output results/tester-rc2
```

The rc1 packet completed an assisted Windows CPU attempt; rc2 still needs its targeted Windows retest. Internet access is needed for dependencies. Retain terminal output. You may run `python scripts/verify_candidate.py` using the environment's Python to check packet identity. GPU is optional.

## Task A — first use and interpretation

Produce a complete report with the supplied independent workload. Locate the selected configuration, gradient/state evidence, eligible rankings and held-out outcome. Explain in your own words what the recommendation establishes, what it does not establish, and how you would react to heldout_failed or no_recommendation. Consult the README/product guide as needed; record what you consulted and where you needed help. Do not edit package internals.

## Task B — integrate your existing workload

If available, use a pre-existing deterministic dynamics function AND scalar objective of your own. Before scheduling, tell the organizer its device/dtype, state shape, named parameters and reference approach; this screens feasibility, not step-by-step implementation. Use examples/product_linear.py as the factory template and the guide as the contract. Supply an analytical or otherwise independently justified differentiable reference, candidates, tolerances and at least one snapshot. Run the CLI and retain its outputs. A failed candidate is valid evidence. Do not invent an entire new model/reference solely for this session.

If no suitable workload is available, record Task B as not attempted; Task A still tests onboarding, but does not establish independent custom-workload reuse. A supplied example with changed parameters is not equivalent to bringing your own model/objective.

## Feedback and evidence

Copy FEEDBACK_TEMPLATE.md and fill it in. Share the completed form, terminal log and the output directories (report/recommendation/provenance/rankings/lock). For custom work, include the adapter and source only if you can share them; remove private data and describe anything withheld. Source paths now default to relative paths or opaque external-file identifiers. Check exported snapshots and caller metadata before sharing; these can still contain private data. Record every intervention, including who helped and when. Assistance is allowed and must remain visible.

Suggested target is30–45minutes after dependency installation; it is not a pass deadline. Stop and report if blocked. The developer will retain the attempt, fix demonstrated friction and ask for a focused retest against a separately identified revision. Do not overwrite first-attempt artifacts.

## Targeted rc2 retest

1. Follow the Windows uv route in a genuinely new environment; record any help required.
2. Run Task A into results/tester-rc2; preserve the original attempt.
3. Inspect report.json, recommendation.json and provenance.json: supplied source paths should contain no home-directory prefix. Confirm raw and canonical hashes exist; run the candidate verifier.
4. Explain heldout_failed and no_recommendation using the updated guide. No failed-outcome run or new research experiment is required.
5. Return the revised feedback form, terminal log and output folder. Task B may remain not attempted if you have no suitable existing workload.
