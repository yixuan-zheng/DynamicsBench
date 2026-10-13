# Public v0.1.0 verification — 2026-10-13

- A fresh virtual environment installed the release wheel with the optional
  upstream-workload dependencies. Python 3.11, PyTorch 2.14.1, NumPy 2.4.6 and
  torchdiffeq 0.2.5 were observed on macOS CPU.
- All 19 CPU tests passed; three CUDA tests skipped because CUDA was unavailable.
- Both CLI examples returned selected with held-out checks passing: analytical
  RK4/4 and upstream short-window Euler/9. Timings are machine-specific.
- Product source under src/ is unchanged from the tested rc2 commit. v0.1.0
  changes distribution, licensing, documentation and selected public evidence.
- Both historical commits were inspected, including 42 distinct blobs and
  embedded wheel contents. The public payload was scanned for private usernames,
  personal home paths and common credential patterns, including tensor archives.
  No matches were found. This is a targeted audit, not a general security proof.
- Scheduler logs, operational session records and private tester correspondence
  were excluded. Six research manifest path labels were made relative; their
  scientific values and source hashes were preserved. All eight frozen scientific
  inputs retained their exact hashes.
- RELEASE_MANIFEST.json records the public payload and declared UTF-8 line-ending
  equivalence. It excludes itself. scripts/verify_candidate.py verifies it.
- CPU GitHub Actions checks are configured for Linux, Windows and macOS. Their
  live execution status is available on the repository Actions page; this local
  record does not pre-claim their result or any new GPU verification.
