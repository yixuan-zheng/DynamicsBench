# DynamicsBench — frozen reserved-seed evaluation protocol v2

Session date: 2026-10-11 (America/New_York). Frozen before any reserved-seed data generation or execution. Exact input hashes are in final-freeze.json; the runner verifies them before generating data and writes PRE_RUN.json. Development history remains in PROTOCOL.md. This document replaces provisional choices with the values exercised in the completed development trials.

## Question and scope

Does checkpoint-aware numerical validation improve training decisions over fixed moderate RK4 after counting validation and training cost? Compare fixed, initialization-only, and initialization-plus-early strategies. Allow mixed or negative findings. No changes to candidates, model, tolerances, optimizer, budgets, or quality criterion after evaluation begins. Research extensions such as budget-aware revalidation remain future options, outside this experiment.

This is one damped nonlinear oscillator task with two frequency regimes and three paired seeds per regime. It is not a claim about arbitrary differential equations, stiff systems, large neural networks, GPU hardware efficiency, or energy consumption.

## Task and data

Teacher: q′=v; v′=−alpha*q−q³−0.15*v. Alpha1 and16; horizon6;16 equally spaced observation times. Model: q′=v exactly, learned acceleration with2→32 tanh→32 tanh→1 MLP. Alpha and teacher coefficients are not provided to the model. Initialization follows examples/development_pilot.py unchanged.

Evaluation model seeds11,12,13. Training32 initial conditions with seed1000+model_seed; calibration8 with2000+model_seed; held-out64 with3000+model_seed. Initial q/v independently uniform[-1,1]. The same model seed/data are paired across all strategies; regimes also share initial conditions. Training targets, calibration targets and held-out targets use float64 RK4/1024, each independently checked against2048: aggregate state L2 error ≤1e−8+1e−5×reference L2 norm. Teacher failure is a failed case, not grounds to retune.

State scales are standard deviations of training teacher trajectories over observations and batch, clamped at1e−6. Scales are recorded per case and shared by strategies. Calibration/held-out states never enter training, warm-start, or scale estimation. Held-out results never affect recommendations or parameter updates.

## Common training and pairing

All float64, eager PyTorch on PACE V100; deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8. Adam lr0.003, default PyTorch Adam settings. Common warm-start:100updates fitting512 consecutive short windows from the32 training trajectories and16 teacher intervals. Horizon0.375, RK4/8, one endpoint observation per window; normalized endpoint MSE. Window starts include training initial states and prior teacher states. Then200 full-horizon updates with normalized trajectory MSE.

Clone both post-warm-start parameters and Adam momentum/step state for every strategy. Initialization in the selection policies means **after this common warm-start**, not random weights. Strategy order rotates: seed11 fixed/initial-only/early; seed12 initial-only/early/fixed; seed13 early/fixed/initial-only. Cases run as independent jobs, at most4concurrent; each paired strategy comparison is sequential on one GPU.

## Strategies and selection

1. Fixed: RK4/64 for200full-horizon updates, no selection cost.
2. Initialization-only: select at post-warm-start initialization; chosen solver for200updates.
3. Initialization-plus-early: first20updates withRK4/64; select using initial AND update20 checkpoints, requiring both to pass; chosen solver for remaining180updates. The20updates replace part of the200budget.

Candidates: Euler/32/64; RK4/16/32/64/128/256. Calibration objective: normalized trajectory MSE against calibration teacher states. Reference RK4/512. Every observation state L2 error must be ≤1e−6+1e−3×reference norm; every named parameter-group gradient L2 error ≤1e−7+1e−2×reference norm. Parameter groups w1/w2/w3/b1/b2/b3. Near-zero-reference threshold1e−10; absolute tolerance still applies. Record relative error and gradient direction in addition to pass/fail.

Choose lowest synchronized median complete forward/backward execution time among passing candidates, using2warmups and3repeats; retain timing ranges/overlaps. If none passes, record selection failure and use explicitRK4/256fallback. A fallback that fails numerical criteria remains a reported failure, not a trusted certification. No late retuning.

## Research diagnostics and reference credibility

Save initial/update20/update100/update200 calibration snapshots. Independently refine modelRK4/512 against1024 at all saved snapshots: state tolerances1e−7absolute/1e−4relative, named gradient tolerances1e−8absolute/1e−3relative. Primary model state checks are per observation. Check final held-out states512vs1024 with aggregate L2 tolerance1e−7+1e−4×reference norm. Reference failures invalidate a case's comparative quality inference and are reported explicitly; no silent exclusion or setting changes.

Final held-out normalized MSE uses a commonRK4/512solver for every trained model. Final chosen configuration is audited at all saved calibration snapshots; initial/early strategy diagnostics therefore do not imply that chosen configuration was used during its first20updates. Late100/200checks do not influence selection. Distinguish numerical violations, false reassurance, training failures, and prediction quality.

## Cost and operational criterion

Primary policy cost: common warm-start charged to each standalone policy, cloned weights/optimizer setup, full-horizon training, complete selection references/candidate checks/timing calls, and selection-required snapshots. Research-only snapshot capture excluded and recorded separately. Teacher generation, reference audits, late checks, held-out evaluation and artifact serialization are research work, outside primary policy cost. Queue delay, module/container startup, and imports are not execution cost. Actual case study wall time is separately retained; sums of charged policy cost intentionally duplicate shared warm-start work.

A selected strategy meets the useful quality–cost criterion if references for it and its paired fixed baseline pass, its total policy seconds are LOWER than fixed, and held-out normalized MSE ≤fixed MSE×1.05+1e−4. This threshold is an operational allowance, not a theorem. Late numerical pass/fail is reported separately, not silently folded into the quality criterion. Always report raw cost/MSE, ratios, selection overhead, fallback use, and each paired seed.

## Completion and failure rules

Six cases ×three strategies =18runs. Require all expected artifacts or explicit failure records. No seed replacement, rerunning solely to improve results, or tuning after inspection. Nonfinite training aborts the affected case and is retained as a failure. Output overwrite is refused. Any necessary software correction after freeze requires an explicit versioned amendment, preservation of original runs, and transparent reporting.

Report descriptive counts/paired results, not statistical significance claims from three seeds. A transient gradient violation without quality loss is distinct from a failed training decision. Passing numerical checks with poorer held-out quality is also retained. A negative checkpoint-benefit result is a valid outcome.
