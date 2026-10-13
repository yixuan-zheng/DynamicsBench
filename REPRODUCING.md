# Reproducing the evidence

## Product

Follow QUICKSTART.md in a fresh environment. Run the supplied analytical and upstream
examples and the product test suite. Runtime ranks are measurements, not constants.

## Study aggregates

Install matplotlib separately, copy evidence/frozen-evaluation to a disposable
directory, then run:

```sh
python research-replay/examples/summarize_evaluation.py COPY_OF_FROZEN_EVALUATION
```

This writes summaries/plots; retain the release evidence unchanged.

## Frozen training replay

The historical study used V100, Python 3.10 and the PACE PyTorch 2.1 container.
This is separate from the current product installation. On a compatible allocated
environment, from research-replay, source-mount `src` and run:

```sh
export PYTHONPATH="$PWD/src"
export APPTAINERENV_PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export APPTAINERENV_OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export APPTAINERENV_CUBLAS_WORKSPACE_CONFIG=:4096:8
python examples/policy_evaluation.py --alpha 1 --seed 11 --output ../reruns/alpha1-seed11 --freeze research/final-freeze.json
```

Repeat alpha 1/16 and seeds 11/12/13 with distinct output directories. The original
runner verifies eight frozen input hashes and refuses existing outputs. The retained
scheduler script documents historical module/job settings; adapt scheduling to your
cluster outside the frozen inputs. Queue access and hardware are not provided.
Different dependencies/hardware can alter numerics and timings.

The supplemental compiled example is a separate replay route, not an exact frozen
training input or a guaranteed cold-cache reproduction. It needs a compatible CUDA
installation and compiler driver discovery. Detailed operational repair histories
remain private; product CLI benchmarking uses eager execution only.
