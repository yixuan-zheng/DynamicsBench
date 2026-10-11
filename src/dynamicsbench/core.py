"""Deterministic, CPU/CUDA validation of user-defined differentiable dynamics."""
from dataclasses import dataclass, asdict
from typing import Callable, Mapping
import hashlib
import math
import platform
import statistics
import time
import torch

TensorMap = Mapping[str, torch.Tensor]
Dynamics = Callable[[float, torch.Tensor, TensorMap], torch.Tensor]
Objective = Callable[[torch.Tensor, TensorMap], torch.Tensor]

@dataclass(frozen=True)
class SolverConfig:
    method: str
    steps: int
    def __post_init__(self):
        if self.method not in ('euler', 'rk4') or type(self.steps) is not int or self.steps < 1:
            raise ValueError('Use euler/rk4 and positive steps')

@dataclass(frozen=True)
class Tolerances:
    state_atol: float = 1e-6
    state_rtol: float = 1e-3
    gradient_atol: float = 1e-7
    gradient_rtol: float = 1e-2
    small_norm: float = 1e-10
    def __post_init__(self):
        if any(not math.isfinite(v) for v in asdict(self).values()) or min(self.state_atol, self.state_rtol, self.gradient_atol, self.gradient_rtol) < 0 or self.small_norm <= 0:
            raise ValueError('Tolerances must be nonnegative; small_norm positive')

class ValidationCase:
    """Own detached snapshots; caller mutation cannot change evaluation inputs."""
    def __init__(self, name: str, initial: torch.Tensor, parameters: TensorMap, horizon: float = 1., observations: int = 1):
        if not isinstance(name,str) or not name or not parameters or not math.isfinite(horizon) or horizon <= 0:
            raise ValueError('Require name, named parameters and positive horizon')
        if type(observations) is not int or observations < 1:
            raise ValueError('Observations must be a positive integer')
        self.observations = observations
        if any(not isinstance(k,str) or not k for k in parameters):
            raise ValueError("Parameter names must be nonempty strings")
        tensors = [initial, *parameters.values()]
        if any(not isinstance(t,torch.Tensor) for t in tensors):
            raise ValueError("Initial state and parameters must be tensors")
        if any(t.device != initial.device or t.dtype != initial.dtype or t.dtype not in (torch.float32, torch.float64) or not torch.isfinite(t).all() for t in tensors):
            raise ValueError('Require finite float32/float64 snapshots on a common device/dtype')
        self.name, self.horizon = name, horizon
        self._initial = initial.detach().clone()
        self._parameters = {k:v.detach().clone() for k,v in sorted(parameters.items())}
    def fresh(self):
        return self._initial.clone(), {k:v.clone().requires_grad_(True) for k,v in self._parameters.items()}
    def manifest(self):
        payload = {'initial':self._initial.tolist(), 'parameters':{k:v.tolist() for k,v in self._parameters.items()}, 'horizon':self.horizon,'observations':self.observations}
        import json
        return {'name':self.name, 'snapshot':payload, 'sha256':hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}

def solve(dynamics: Dynamics, z: torch.Tensor, parameters: TensorMap, horizon: float, config: SolverConfig):
    return _solve_steps(dynamics,z,parameters,horizon,config.method,config.steps)

def _solve_steps(dynamics,z,parameters,horizon,method,steps):
    h = horizon/steps
    for i in range(steps):
        t = i*h
        k1 = dynamics(t, z, parameters)
        if method == 'euler':
            z = z+h*k1
        else:
            k2 = dynamics(t+h/2, z+h*k1/2, parameters)
            k3 = dynamics(t+h/2, z+h*k2/2, parameters)
            k4 = dynamics(t+h, z+h*k3, parameters)
            z = z+h*(k1+2*k2+2*k3+k4)/6
    return z

def solve_observed(dynamics, z, parameters, horizon, config, observations):
    """Return states at equally spaced times, preserving global-time dynamics."""
    if observations < 1 or config.steps % observations:
        raise ValueError('Steps must be divisible by observations')
    segment=horizon/observations
    states=[]
    for i in range(observations):
        offset=i*segment
        z=_solve_steps(lambda t,z,p:dynamics(t+offset,z,p),z,parameters,segment,
                config.method,config.steps//observations)
        states.append(z)
    return torch.stack(states)

def execute(case, dynamics, objective, config, reference=None):
    z, p = case.fresh()
    if reference:
        state = reference(z,p,case.horizon)
    elif case.observations > 1:
        state = solve_observed(dynamics,z,p,case.horizon,config,case.observations)
    else:
        state = solve(dynamics,z,p,case.horizon,config)
    expected=(case.observations,*z.shape) if case.observations>1 else z.shape
    if not isinstance(state,torch.Tensor) or state.shape != expected:
        raise ValueError('Dynamics must preserve state shape')
    loss = objective(state,p)
    if not isinstance(loss,torch.Tensor) or loss.ndim != 0:
        raise ValueError('Objective must return a scalar tensor')
    values = torch.autograd.grad(loss, tuple(p.values()), allow_unused=True) if loss.requires_grad else (None,)*len(p)
    grads = {k:torch.zeros_like(p[k]) if g is None else g.detach() for k,g in zip(p,values)}
    return state.detach(), grads

def metrics(actual, reference, atol, rtol, small):
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        return {'passed':False, 'failure':'nonfinite', 'absolute_l2':None, 'relative_l2':None, 'cosine':None}
    a, b = actual.flatten(), reference.flatten()
    err, norm = torch.linalg.vector_norm(a-b).item(), torch.linalg.vector_norm(b).item()
    anorm = torch.linalg.vector_norm(a).item()
    return {'passed':err <= atol+rtol*norm, 'absolute_l2':err,
            'relative_l2':None if norm < small else err/norm,
            'cosine':None if min(norm,anorm)<small else max(-1.,min(1.,torch.dot(a,b).item()/(norm*anorm))),
            'reference_l2':norm, 'limit':atol+rtol*norm, 'small_reference':norm<small}

def compare(case, dynamics, objective, config, ref, tolerances):
    try:
        state, grads = execute(case,dynamics,objective,config)
        sm = metrics(state, ref[0], tolerances.state_atol,tolerances.state_rtol,tolerances.small_norm)
        trajectory = [metrics(a,b,tolerances.state_atol,tolerances.state_rtol,tolerances.small_norm) for a,b in zip(state,ref[0])] if case.observations>1 else []
        gm = {k:metrics(g,ref[1][k],tolerances.gradient_atol,tolerances.gradient_rtol,tolerances.small_norm) for k,g in grads.items()}
        return {'case':case.name, 'state':sm,'trajectory_observations':trajectory, 'gradients':gm, 'passed':(all(m['passed'] for m in trajectory) if trajectory else sm['passed']) and all(m['passed'] for m in gm.values())}
    except (RuntimeError, ValueError) as e:
        return {'case':case.name, 'passed':False, 'failure':type(e).__name__+': '+str(e)}

def evaluate(dynamics: Dynamics, objective: Objective, selection_cases, candidates, *, reference: Callable, reference_id: str, workload_id: str, heldout_cases=(), tolerances=Tolerances(), repeats=7):
    """Reference is user-supplied; its credibility must be independently established."""
    selection_cases, heldout_cases, candidates = list(selection_cases), list(heldout_cases), list(candidates)
    if not selection_cases or not candidates or type(repeats) is not int or repeats < 3:
        raise ValueError('Require selection cases, candidates and at least three repeats')
    cases = selection_cases+heldout_cases
    if len({c.name for c in cases}) != len(cases):
        raise ValueError('Case names must be unique across selection and heldout')
    device = cases[0]._initial.device
    if device.type not in ('cpu', 'cuda') or any(c._initial.device != device or c._initial.dtype != cases[0]._initial.dtype for c in cases):
        raise ValueError('Require common CPU/CUDA device and dtype across cases')
    def synchronize():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    refs = {c.name:execute(c,dynamics,objective,candidates[0],reference) for c in cases}
    if any(not torch.isfinite(s).all() or any(not torch.isfinite(g).all() for g in gs.values()) for s,gs in refs.values()):
        raise ValueError('Reference contains nonfinite state or gradients')
    rows=[]
    for config in candidates:
        checks=[compare(c,dynamics,objective,config,refs[c.name],tolerances) for c in selection_cases]
        row={'configuration':asdict(config),'checks':checks,'passed':all(c['passed'] for c in checks)}
        if row['passed']:
            for _ in range(2):
                for c in selection_cases: execute(c,dynamics,objective,config)
            synchronize()
            if device.type == 'cuda':
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            timings=[]
            for _ in range(repeats):
                synchronize()
                start=time.perf_counter()
                for c in selection_cases: execute(c,dynamics,objective,config)
                synchronize()
                timings.append((time.perf_counter()-start)*1000)
            row['timing_ms']={'median':statistics.median(timings),'min':min(timings),'max':max(timings),'samples':timings}
            if device.type == 'cuda':
                row['memory_bytes'] = {'baseline_allocated':baseline,'peak_allocated':torch.cuda.max_memory_allocated(device),'incremental_peak_allocated':max(0,torch.cuda.max_memory_allocated(device)-baseline)}
        rows.append(row)
    passing=[r for r in rows if r['passed']]
    selected=min(passing,key=lambda r:r['timing_ms']['median']) if passing else None
    selected_config=SolverConfig(**selected['configuration']) if selected else None
    heldout=[compare(c,dynamics,objective,selected_config,refs[c.name],tolerances) for c in heldout_cases] if selected else []
    ambiguous=[]
    if selected:
        ambiguous=[r['configuration'] for r in passing if r is not selected and r['timing_ms']['min'] <= selected['timing_ms']['max'] and selected['timing_ms']['min'] <= r['timing_ms']['max']]
    return {'schema_version':2,'manifest':{'python':platform.python_version(),'pytorch':torch.__version__,'platform':platform.platform(),'device':str(device),'dtype':str(cases[0]._initial.dtype),'cuda_runtime':torch.version.cuda,'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),'torch_threads':torch.get_num_threads(),'workload_id':workload_id,'reference_id':reference_id,'cases':[c.manifest() for c in cases],'selection_cases':[c.name for c in selection_cases],'heldout_cases':[c.name for c in heldout_cases],'timing':'synchronized wall-clock; fresh snapshot + complete forward/backward; 2 warmups','repeats':repeats},'tolerances':asdict(tolerances),'results':rows,'selected_configuration':asdict(selected_config) if selected else None,'timing_overlap_candidates':ambiguous,'heldout_checks':heldout,'heldout_passed':all(c['passed'] for c in heldout) if heldout else None,'scope':'Selection excludes heldout cases. No guarantee beyond tested snapshots; overlapping timing ranges mean speed ranking is uncertain.'}
