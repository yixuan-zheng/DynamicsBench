# Verified profiles

Clean Mac CPU: Python3.11.5, PyTorch2.14.1, NumPy2.4.6, torchdiffeq0.2.5, SciPy1.17.1. The base package declares Python>=3.11, Torch>=2.2 and NumPy>=1.26; these minimums are not an assertion that every combination was tested.

PACE GPU: A40, module pytorch/25.10, Python3.12.3, Torch2.9.0a0+145a3a7bda.nv25.10/CUDA13.0. Installed wheel, no source mount. All36 tests passed after job-local CUDA-library discovery setup for Triton. The newer module cannot execute on V100 (capability7.0 versus minimum7.5). Compiled checks need additional module-specific setup; CPU is the usability default.

Historical V100 studies used Python3.10/Torch2.1 source execution. That is not a supported installation claim for this package. Full cluster setup and evidence remain in the author's archive; this packet deliberately contains no account or cluster configuration.
