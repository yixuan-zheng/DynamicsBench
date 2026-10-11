"""Public-API adapter for the pinned upstream torchdiffeq neural-ODE demo."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from torchdiffeq import odeint
from dynamicsbench.core import ValidationCase, SolverConfig, execute, evaluate, metrics

ROOT=Path(__file__).resolve().parents[1]
OBS=9
HORIZON=25./999*9  # Upstream defaults: data_size1000, batch_time10.


def load_upstream_class():
    file=ROOT/'external/torchdiffeq/odefunc.py'
    spec=importlib.util.spec_from_file_location('upstream_odefunc',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ODEFunc


def make_workload(device='cpu'):
    device=torch.device(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        module=load_upstream_class()().to(device=device,dtype=torch.float64)
    parameters={name:p.detach().clone() for name,p in module.named_parameters()}
    def dynamics(time,state,p):
        return torch.func.functional_call(module,p,(time,state),strict=True)
    matrix=torch.tensor([[-.1,2.],[-2.,-.1]],device=device,dtype=torch.float64)
    def teacher(time,state):return torch.mm(state**3,matrix)
    origin=torch.tensor([[2.,0.]],device=device,dtype=torch.float64)
    starts=torch.tensor([0.,2.,5.],device=device,dtype=torch.float64)
    grid=torch.linspace(0.,HORIZON,OBS+1,device=device,dtype=torch.float64)
    with torch.no_grad():
        initial=odeint(teacher,origin,starts,rtol=1e-12,atol=1e-14).squeeze(1)
        coarse_initial=odeint(teacher,origin,starts,rtol=1e-10,atol=1e-12).squeeze(1)
        target=odeint(teacher,initial,grid,rtol=1e-12,atol=1e-14)
        coarse=odeint(teacher,coarse_initial,grid,rtol=1e-10,atol=1e-12)
    teacher_checks=[metrics(a,b,1e-8,1e-6,1e-10) for a,b in zip(coarse,target)]
    if not all(m['passed'] for m in teacher_checks):raise RuntimeError('Independent teacher refinement failed')
    def objective(states,p):
        # Restore upstream t=0 term, which contributes zero parameter gradient.
        pred_y=torch.cat((initial.unsqueeze(0),states),dim=0)
        batch_y=target
        return torch.mean(torch.abs(pred_y-batch_y))
    def reference(z,p,h,rtol=1e-9,atol=1e-11):
        times=torch.linspace(0.,h,OBS+1,device=z.device,dtype=z.dtype)
        return odeint(lambda t,y:dynamics(t,y,p),z,times,method='dopri5',rtol=rtol,atol=atol)[1:]
    perturbed={k:v.clone() for k,v in parameters.items()}
    perturbed['net.2.weight']=perturbed['net.2.weight']*1.1
    cases=[ValidationCase('upstream_initialization',initial,parameters,HORIZON,OBS),
           ValidationCase('parameter_perturbation',initial,perturbed,HORIZON,OBS)]
    return {'module':module,'dynamics':dynamics,'objective':objective,'reference':reference,'cases':cases,
        'target':target,'teacher_checks':teacher_checks,'initial':initial,'parameters':parameters}


def run(output,device='cpu'):
    output.mkdir(parents=True,exist_ok=True)
    workload=make_workload(device)
    dynamics=workload['dynamics'];objective=workload['objective'];reference=workload['reference']
    before={k:v.detach().clone() for k,v in workload['module'].named_parameters()}
    reference_checks=[]
    for case in workload['cases']:
        coarse=execute(case,dynamics,objective,SolverConfig('rk4',9),reference)
        fine=execute(case,dynamics,objective,SolverConfig('rk4',9),lambda z,p,h:reference(z,p,h,1e-11,1e-13))
        state_checks=[metrics(a,b,1e-8,1e-6,1e-10) for a,b in zip(coarse[0],fine[0])]
        gradients={k:metrics(g,fine[1][k],1e-9,1e-5,1e-10) for k,g in coarse[1].items()}
        entry={'case':case.name,'passed':all(m['passed'] for m in state_checks+list(gradients.values())),
            'states':state_checks,'gradients':gradients,
            'minimum_noninitial_abs_residual':(fine[0]-workload['target'][1:]).abs().min().item()}
        reference_checks.append(entry)
    if not all(c['passed'] for c in reference_checks):raise RuntimeError('Independent model/gradient reference refinement failed')
    candidates=[SolverConfig('euler',n) for n in (9,18,36,72)]+[SolverConfig('rk4',n) for n in (9,18,36,72)]
    report=evaluate(dynamics,objective,workload['cases'][:1],candidates,
        reference=reference,reference_id='torchdiffeq0.2.5 DOPRI5 rtol1e-9 atol1e-11; refined1e-11/1e-13',
        workload_id='pinned-torchdiffeq-ode-demo-L1',heldout_cases=workload['cases'][1:],repeats=5)
    for k,v in workload['module'].named_parameters():
        if not torch.equal(v,before[k]):raise RuntimeError('Module parameters mutated by functional adapter')
    provenance=json.loads((ROOT/'external/torchdiffeq/provenance.json').read_text())
    report['independent_integration']={'provenance':provenance,'core_changes':False,'teacher_checks':workload['teacher_checks'],
        'reference_checks':reference_checks,'module_weights_unchanged':True,'external_human_usability_test':False,
        'adaptations':['Extract upstream class to avoid argument parsing/data generation on import','Bind explicit parameter snapshots with functional_call(strict=True)',
            'Use three deterministic phases of upstream cubic teacher instead of random mini-batches','Float64 short batch window; retain t=0 term in upstream L1 expression',
            'Second case is an explicit weight perturbation, not a trained checkpoint'],
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'external/torchdiffeq/ode_demo.py',ROOT/'external/torchdiffeq/odefunc.py']}}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'recommendation.json').write_text(json.dumps({'configuration':report['selected_configuration'],'workload_id':report['manifest']['workload_id'],
        'heldout_passed':report['heldout_passed'],'scope':'Only recorded checkpoints, objective, tolerances and environment; no training-quality claim.'},indent=2)+'\n')
    torch.save({'snapshots':[c.manifest() for c in workload['cases']],'target':workload['target'].cpu()},output/'workload.pt')
    print('Independent integration:',report['selected_configuration'],'perturbation check:',report['heldout_passed'],'reference checks:',all(c['passed'] for c in reference_checks))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    parser.add_argument('--output',type=Path,default=Path('results/product/p1/local'))
    args=parser.parse_args();run(args.output,args.device)
