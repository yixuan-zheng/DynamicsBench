"""Batched CUDA validation with analytical gradients and a profiler trace."""
import argparse
import json
from pathlib import Path
import torch
from dynamicsbench.core import ValidationCase, SolverConfig, evaluate, execute

def dynamics(t,z,p): return p['rates']*z

def objective(z,p): return z.square().mean()/2

def reference(z,p,h): return z*torch.exp(p['rates']*h)

def main(output):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    output.mkdir(parents=True,exist_ok=True)
    configs=[SolverConfig('euler',128),SolverConfig('rk4',8),SolverConfig('rk4',32)]
    reports=[]
    for batch in (32,512):
        for dtype in (torch.float32,torch.float64):
            initial=torch.ones(batch,128,device='cuda',dtype=dtype)
            rates=torch.linspace(-.3,-.1,128,device='cuda',dtype=dtype)
            case=ValidationCase(f'batch{batch}_{dtype}',initial,{'rates':rates})
            report=evaluate(dynamics,objective,[case],configs,reference=reference,
                reference_id='analytical diagonal exponential',workload_id=f'batch{batch}_state128',repeats=9)
            reports.append(report)
            print(batch,dtype,report['selected_configuration'])
    (output/'reports.json').write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        execute(case,dynamics,objective,SolverConfig('rk4',8))
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output/'trace.json'))
    (output/'profiler.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=20))
    print('GPU:',torch.cuda.get_device_name())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/pace/milestone3'));main(p.parse_args().output)
