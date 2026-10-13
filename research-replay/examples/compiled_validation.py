import argparse
import json
from pathlib import Path
import torch
from dynamicsbench.core import ValidationCase,SolverConfig
from dynamicsbench.execution import compare_execution,Runner
from gpu_validation import dynamics,objective,reference

def main(output):
    output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for batch in (32,512):
        for dtype in (torch.float32,torch.float64):
            case=ValidationCase(f'b{batch}_{dtype}',torch.ones(batch,128,device='cuda',dtype=dtype),{'rates':torch.linspace(-.3,-.1,128,device='cuda',dtype=dtype)})
            r=compare_execution(case,dynamics,objective,SolverConfig('rk4',8),reference)
            reports.append(r)
            (output/'compiled-reports.json').write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
            print(case.name,r['steady_state_speedup'],r['timing_ranges_overlap'],flush=True)
    runner=Runner(dynamics,objective,SolverConfig('rk4',8),'compiled')
    for _ in range(3):runner(case)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        runner(case);torch.cuda.synchronize()
    prof.export_chrome_trace(str(output/'compiled-trace.json'))
    (output/'compiled-profiler.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=20))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/pace/milestone3/compiled'));main(p.parse_args().output)
