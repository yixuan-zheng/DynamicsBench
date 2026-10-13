"""Validated eager/compiled execution comparison, independent of solver selection."""
import platform
import statistics
import time
from dataclasses import asdict
import torch
from .core import execute, metrics, solve, solve_observed, Tolerances

class Runner:
    def __init__(self, dynamics, objective, config, backend='eager'):
        if backend not in ('eager','compiled'): raise ValueError('Unknown backend')
        def forward(z,p,h,observations):
            state=solve_observed(dynamics,z,p,h,config,observations) if observations>1 else solve(dynamics,z,p,h,config)
            return state,objective(state,p)
        self.forward=torch.compile(forward,fullgraph=True) if backend=='compiled' else forward
    def __call__(self,case):
        z,p=case.fresh()
        state,loss=self.forward(z,p,case.horizon,case.observations)
        expected=(case.observations,*z.shape) if case.observations>1 else z.shape
        if state.shape!=expected or loss.ndim!=0: raise ValueError('Invalid state/objective shape')
        values=torch.autograd.grad(loss,tuple(p.values()),allow_unused=True) if loss.requires_grad else (None,)*len(p)
        return state.detach(),{k:torch.zeros_like(p[k]) if g is None else g.detach() for k,g in zip(p,values)}

def compare_execution(case,dynamics,objective,config,reference,*,repeats=11,tolerances=Tolerances()):
    device=case._initial.device
    if device.type!='cuda' or repeats<3: raise ValueError('CUDA and at least 3 repeats required')
    def sync(): torch.cuda.synchronize(device)
    ref=execute(case,dynamics,objective,config,reference)
    def validation(result):
        state=metrics(result[0],ref[0],tolerances.state_atol,tolerances.state_rtol,tolerances.small_norm)
        grads={k:metrics(g,ref[1][k],tolerances.gradient_atol,tolerances.gradient_rtol,tolerances.small_norm) for k,g in result[1].items()}
        return {'state':state,'gradients':grads,'passed':state['passed'] and all(x['passed'] for x in grads.values())}
    rows={}; runners={}
    for backend in ('eager','compiled'):
        try:
            runners[backend]=Runner(dynamics,objective,config,backend)
            sync();start=time.perf_counter();result=runners[backend](case);sync()
            first_ms=(time.perf_counter()-start)*1000
            check=validation(result);del result
            rows[backend]={'validation':check,'first_call_ms':first_ms}
            if not check['passed']: continue
            for _ in range(3): runners[backend](case)
            sync()
        except Exception as e:
            rows[backend]={'failure':type(e).__name__+': '+str(e),'validation':{'passed':False}}
    timings={k:[] for k,r in rows.items() if r['validation']['passed']}
    peaks={k:[] for k in timings}
    # Alternate backend order each repeat to reduce ordering bias.
    for i in range(repeats):
        order=list(timings) if i%2==0 else list(reversed(timings))
        for backend in order:
            sync();baseline=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
            start=time.perf_counter();result=runners[backend](case);sync()
            timings[backend].append((time.perf_counter()-start)*1000)
            peaks[backend].append(max(0,torch.cuda.max_memory_allocated(device)-baseline));del result
    for backend,samples in timings.items():
        rows[backend]['timing_ms']={'median':statistics.median(samples),'min':min(samples),'max':max(samples),'samples':samples}
        rows[backend]['incremental_peak_allocated_bytes']=max(peaks[backend])
    speedup=None;overlap=None;break_even=None
    if len(timings)==2:
        e,c=rows['eager'],rows['compiled'];em,cm=e['timing_ms']['median'],c['timing_ms']['median']
        speedup=em/cm
        overlap=max(min(timings['eager']),min(timings['compiled']))<=min(max(timings['eager']),max(timings['compiled']))
        import math
        if em>cm: break_even=math.ceil(max(0,c['first_call_ms']-e['first_call_ms'])/(em-cm))
    return {'manifest':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device),'dtype':str(case._initial.dtype),'case':case.manifest(),'solver':asdict(config),'compile':'torch.compile fullgraph=True, default backend/mode','timing':'synchronized wall time; fresh snapshots + forward + backward; 3 warmups; alternating order','cache_policy':'first invocation in this process; persistent compiler cache may exist; NOT guaranteed cold-cache'},'tolerances':asdict(tolerances),'backends':rows,'steady_state_speedup':speedup,'timing_ranges_overlap':overlap,'estimated_amortization_calls':break_even}
