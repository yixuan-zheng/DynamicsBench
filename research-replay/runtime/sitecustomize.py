"""Opt-in compatibility shim for Triton 2.1 in PACE's read-only container.

Use the driver mapped by Apptainer --nv, not a CUDA stub. No system files
are changed. Modern environments should leave DYNAMICSBENCH_DRIVER_DIR unset.
"""
import os
from pathlib import Path

folder=os.environ.get('DYNAMICSBENCH_DRIVER_DIR')
if folder:
    mapped=Path('/.singularity.d/libs/libcuda.so.1')
    if not mapped.exists():
        raise RuntimeError('Apptainer-mapped CUDA driver is missing')
    directory=Path(folder)
    directory.mkdir(parents=True,exist_ok=True)
    link=directory/'libcuda.so'
    if not link.exists():link.symlink_to(mapped)
    import triton.common.build
    # Old Triton searches only ldconfig, whose entries are stale in this image.
    triton.common.build.libcuda_dirs=lambda:[str(directory)]
