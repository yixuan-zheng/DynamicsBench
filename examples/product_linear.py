"""Minimal complete user workload with an analytical differentiable reference."""
from pathlib import Path
import torch
from dynamicsbench import Workload, ValidationCase, SolverConfig

def build_workload(device='cpu'):
    z=torch.ones(2,dtype=torch.float64,device=device)
    def dynamics(t,state,p):return p['rate']*state
    def objective(states,p):return .5*states.square().sum()
    def reference(initial,p,horizon):return initial*torch.exp(p['rate']*horizon)
    return Workload(dynamics,objective,reference,
        [ValidationCase('initial',z,{'rate':-z})],
        [SolverConfig(method,n) for method in ('euler','rk4') for n in (4,16,64)],
        'linear-two-state-v1','analytical exponential',
        heldout_cases=[ValidationCase('shifted',z,{'rate':-1.1*z})],source_files=[Path(__file__)])
