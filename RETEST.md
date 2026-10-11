# rc2 changes and targeted retest

New version/tag: 0.1.0rc2 / p3-v0.1.0rc2. The rc1 tag and first tester evidence remain unchanged.

- Optional Windows user-space uv instructions and explicit environment selection avoid relying on pip inside a minimal venv.
- Default source paths are portable relative labels; external files receive content-based identifiers. Absolute paths require explicit opt-in.
- LF Git checkout policy plus raw/canonical source hashes and candidate verifier distinguish CRLF differences from content changes.
- Outcome guidance avoids treating held-out numerical failure as proof of overfitting, or automatically prescribing RK4.

Follow the targeted rc2 retest in TESTING.md. Use a fresh environment/output directory, verify provenance, explain outcomes and record help. No invented Task B workload is requested. These repairs have automated/local verification; external Windows retest remains pending.
