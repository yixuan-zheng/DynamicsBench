# Public evidence index

| Evidence | Location | Scope |
|---|---|---|
| Frozen protocol and exact 8-input guard | research-replay/research/ | One task, two regimes, three paired seeds |
| Per-case checks/costs and aggregates | evidence/frozen-evaluation/ | 18 completed policies; descriptive comparisons |
| Original tensor snapshots | evidence/frozen-evaluation/alpha*/ | Owner-generated experiment data; load only trusted tensors |
| Separate compilation measurements | evidence/compiled/compiled-reports.json | Analytical V100 workload; not training speedup |
| Upstream workload | examples/independent_workload.py; external/torchdiffeq/ | Pinned MIT source and extraction provenance |
| Windows supplied-example usability | evidence/USABILITY_RECORD.md | One tester; custom workload not attempted |
| Dependency profiles | docs/COMPATIBILITY.md | Recorded observations; minima not fully tested |
| Public payload integrity | RELEASE_MANIFEST.json | SHA-256 plus declared text line-ending equivalence |

Historical private candidate tags remain for provenance. The public v0.1.0 version
changes packaging/documentation/licensing without changing product solver code.
Private tester correspondence, terminal histories, operational session summaries
and scheduler logs are excluded from this public payload.
Six research manifest source-path labels were made relative for public distribution;
their numerical contents and source SHA-256 values were preserved. Tensor data,
policy results and the eight frozen scientific inputs are byte-identical to the
local archive. The full original archive remains private.
