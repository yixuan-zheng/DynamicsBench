# Evidence and scope

Before packaging: clean CPU installation and33 tests passed with3 CUDA skips; installed A40 profile passed36 tests including compiled float32/float64 and trajectory correctness. Independent neural-ODE API/CLI matched snapshots and every numerical check. Its adaptive reference had trajectory and gradient refinement evidence in P1.

The frozen research study ran18 training runs across six cases. Initialization-only selection saved18–22% in easier cases with similar prediction quality. In harder cases selected settings cost2.14–2.31x fixed settings without prediction-quality improvement. Added early-checkpoint selection provided no incremental benefit under the frozen criterion. No new research result is claimed by packaging this candidate.

Full source snapshots, raw runs, protocols, environment locks and job records remain in the author's archive and are not needed for the tester's installation. The short independent example is integration evidence, not a long-horizon training benchmark. Human usability remains pending.
