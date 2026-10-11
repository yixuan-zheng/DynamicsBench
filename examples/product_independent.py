"""Product factory wrapping the unchanged P1 external integration adapter."""
import importlib.util
from pathlib import Path
from dynamicsbench import Workload, SolverConfig

ROOT=Path(__file__).resolve().parents[1]

def build_workload(device='cpu'):
    path=ROOT/'examples/independent_workload.py'
    spec=importlib.util.spec_from_file_location('p1_adapter',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    w=adapter.make_workload(device)
    return Workload(w['dynamics'],w['objective'],w['reference'],w['cases'][:1],
        [SolverConfig(method,n) for method in ('euler','rk4') for n in (9,18,36,72)],
        'pinned-torchdiffeq-ode-demo-L1','torchdiffeq0.2.5 DOPRI5 rtol1e-9 atol1e-11',
        heldout_cases=w['cases'][1:],source_files=[Path(__file__),path,*[ROOT/'external/torchdiffeq'/name for name in ('ode_demo.py','odefunc.py','provenance.json','LICENSE')]],
        metadata={'reference_evidence':'P1 local and PACE trajectory/gradient refinement reports; supplied reference remains caller responsibility',
                  'window':'upstream default short batch; weight perturbation held-out, not trained checkpoint'})
