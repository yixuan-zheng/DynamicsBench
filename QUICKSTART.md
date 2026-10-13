# Installation and first run

Clone the repository and run commands from its root. Python >=3.11 is required.
No GPU is required. These commands install the source as a normal package.

## macOS / Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install ".[independent]"
.venv/bin/dynamicsbench --workload examples/product_linear.py --output results/first-run
.venv/bin/dynamicsbench --workload examples/product_independent.py --output results/upstream-run
.venv/bin/python -m unittest discover -s tests
```

## Windows / user-space uv

Install uv using its official instructions: https://docs.astral.sh/uv/getting-started/installation/
In PowerShell from the repository root:

```powershell
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe ".[independent]"
.venv\Scripts\dynamicsbench.exe --workload examples/product_linear.py --output results/first-run
.venv\Scripts\dynamicsbench.exe --workload examples/product_independent.py --output results/upstream-run
.venv\Scripts\python.exe -m unittest discover -s tests
```

uv can provision user-space Python without global installation/admin rights.
`uv venv` need not include pip; use `uv pip install --python` directly as shown.
Network or institutional policies can still restrict downloads.

## Release wheel

Use the v0.1.0 release wheel instead of `.` in the installation commands:
`dist/dynamicsbench-0.1.0-py3-none-any.whl[independent]`.
Retain the repository examples and external directory; they are not inside the wheel.
Run `python scripts/verify_candidate.py` with your chosen environment to verify
the release payload; UTF-8 CRLF-to-LF equivalence is explicitly recorded.

## Python API

```python
from dynamicsbench import load_workload, run_workload
workload = load_workload("examples/product_linear.py")
report = run_workload(workload, "results/api-first-run", repeats=3)
```

Use new output directories. Read `recommendation.json` and `report.json` together.
CLI exit 0 means selected, 3 means no_recommendation/heldout_failed with artifacts,
and 2 means setup/input/reference failure. Passing held-out checks applies only to
the recorded cases. See docs/PRODUCT_GUIDE.md for exact fields and error handling.
