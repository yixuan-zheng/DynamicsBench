# Product API, CLI and exports

## Workload factory

A trusted Python module defines `build_workload(device='cpu')` and returns `dynamicsbench.Workload`. The CLI takes `--workload FILE --output DIRECTORY`, optional `--factory NAME`, `--device cpu|cuda[:index]`, and `--repeats N` (at least three). Python callers can construct Workload directly or use load_workload. Installation does not bundle the external examples/data: retain the project/example files for these demonstrations.

Workload fields: dynamics, objective, reference, selection_cases, candidates, workload_id and reference_id; optional heldout_cases, Tolerances, source_files and JSON-compatible metadata. Source paths must exist. Workload.source_root optionally sets the base for portable source labels; the loader defaults to the nearest project pyproject.toml directory, otherwise the factory directory. Files outside that root receive content-based opaque identifiers. Record reference justification, datasets and preprocessing explicitly in metadata and retained files.

## Callback and snapshot contract

- Dynamics `(time, state, named_parameters)` returns a state derivative preserving shape. CPU/CUDA float32/float64 finite snapshots share device/dtype across cases. Parameters are explicit tensors with nonempty string names. Cases clone their inputs.
- Objective `(states, named_parameters)` returns a scalar tensor. Gradients are evaluated with respect to named parameters, with unused parameters reported as zeros. Objective closures must not silently use external trainable weights.
- Reference `(initial, named_parameters, horizon)` returns matching states while retaining parameter differentiation. A supplied reference is not automatically ground truth; establish analytical/refinement evidence separately.
- One observation means a terminal tensor shaped like initial. More than one means `(observations, *initial.shape)` at equally spaced times excluding t=0. Candidate steps must be divisible by observations. Invalid combinations appear as candidate failures and can yield no recommendation.
- One objective callback serves every case. If targets differ by case, explicitly bind them in a designed workload representation or run separate evaluations. The P1 adapter uses the same initial state/target at both weight snapshots and restores t=0 in its upstream objective.
- For native nn.Module models, use strict torch.func.functional_call with supplied parameters. Stochastic layers, mutable buffers, randomness and stateful callbacks are outside the verified deterministic contract.

## Interpreting outputs

`report.json` contains schema_version2 core evidence plus a product block: outcome, eligible ranking, file hashes, metadata and workflow elapsed time. State/gradient metrics, failed-candidate reasons, reference identity, snapshot values/hashes, tolerances, timing samples, CUDA memory where available and timing overlaps are retained. `rankings.csv` contains only passing candidates sorted by median execution cost; a header-only file means none passed.

`recommendation.json` schema_version1 is standalone: selected configuration or null, status, usable_on_tested_cases, held-out result, snapshot manifests, tolerances, workload/reference identity, environment, source hashes, timing ambiguity and scope. When status is heldout_failed, the selection-case winner remains visible for diagnosis but usable_on_tested_cases is false. An absent held-out set appears as null, not a claimed held-out pass.

`provenance.json` records explicitly supplied source-file hashes, installed distributions and caller metadata. Paths default to relative POSIX labels, not absolute workstation locations. Raw sha256 identifies exact bytes; canonical_sha256 normalizes CRLF to LF for UTF-8 text while preserving all other content. Binary hashes remain raw. A canonical match tolerates line endings, not code changes. `environment-lock.txt` records distributions. Neither captures every imported dependency, input dataset, native library, closure or reference credibility automatically. Preserve source/data separately; Use CLI --absolute-source-paths or API source_paths="absolute" only when you explicitly want workstation paths for private debugging. Caller metadata, exception text and tensor snapshots are not automatically anonymized; inspect them before sharing.

Execution timing includes fresh snapshot construction and forward/backward; two warmups precede repeated synchronized wall-clock samples. CUDA memory is allocated tensor memory, not total process/device consumption. Workflow elapsed time additionally includes reference calculation and checks, but excludes factory setup and exports. Timing dispersion is reported; repeat timing or the chosen winner can differ when ranges overlap.

## Outcomes and errors

CLI exit0: selected; exit3: no_recommendation or heldout_failed, with complete artifacts retained; exit2: input/setup/reference error, with an actionable message and no completed report. Candidate numerical failures are evidence, not setup exceptions. heldout_failed means the selected setting failed a recorded held-out numerical check; it does not prove overfitting or universal deployment unsuitability. Inspect which states/gradients failed and the reference/failure reason before choosing finer steps or another solver. no_recommendation means no tested candidate passed; inspect invalid grids, nonfinite states and callback/reference issues as well as tolerances. Reference failures may instead abort with exit2. Missing CUDA prompts CPU or an allocated GPU with a CUDA-enabled installation. The output directory must be new or empty; successful runs never silently overwrite previous evidence.

Repeat runs should reproduce callback inputs, snapshots and numerical outcomes within stated tolerances; execution times are not expected to be identical. The focused tests compare API/CLI cases and checks rather than requiring identical timing ranks.

The product CLI evaluates eager solver configurations only. Separate analytical compilation evidence is in evidence/compiled and TECHNICAL_REPORT.md.
