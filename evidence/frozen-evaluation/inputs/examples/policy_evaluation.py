"""Frozen paired evaluation; only reserved seeds11–13 are accepted."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import torch
from development_pilot import teacher, make_initial, params, acceleration_model, OBS, H
from dynamicsbench.core import SolverConfig, ValidationCase, solve_observed, execute, compare, evaluate, Tolerances

STRATEGIES=('fixed','initial_only','initial_plus_early')
CANDIDATES=[SolverConfig('euler',32),SolverConfig('euler',64)]+[SolverConfig('rk4',n) for n in (16,32,64,128,256)]

def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()

def refined_teacher(alpha,z):
    with torch.no_grad():
        target=teacher(alpha,z,1024);fine=teacher(alpha,z,2048)
    error=(target-fine).norm().item();limit=1e-8+1e-5*fine.norm().item()
    if error>limit:raise RuntimeError('Teacher refinement failed')
    return target,{'absolute_l2':error,'limit':limit,'passed':True}

def warm_start(z,target,scale,p):
    optimizer=torch.optim.Adam(list(p.values()),lr=.003)
    starts=torch.cat((z.unsqueeze(0),target[:-1]),dim=0).reshape(-1,2).detach()
    ends=target.reshape(-1,2).detach();history=[]
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        predicted=solve_observed(acceleration_model,starts,p,H/OBS,SolverConfig('rk4',8),1)
        loss=((predicted-ends)/scale).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite warm-start loss')
        loss.backward();optimizer.step();history.append(loss.item())
    return optimizer,history

def verify_freeze(path,root,alpha,seed):
    if seed not in (11,12,13) or alpha not in (1,16):raise ValueError('Reserved evaluation cases only')
    freeze=json.loads(path.read_text())
    if freeze['evaluation_seeds']!=[11,12,13] or freeze['alphas']!=[1,16]:raise ValueError('Unexpected frozen experiment matrix')
    for relative,expected in freeze['files'].items():
        actual=hashlib.sha256((root/relative).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Frozen input changed: {relative}')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(alpha,seed,output,freeze_path=Path('research/final-freeze.json')):
    frozen_hash=verify_freeze(freeze_path,Path.cwd(),alpha,seed)
    if (output/'policies.json').exists() or (output/'PRE_RUN.json').exists():raise ValueError('Evaluation output already exists; refusing to overwrite')
    output.mkdir(parents=True,exist_ok=True)
    (output/'PRE_RUN.json').write_text(json.dumps({'freeze_sha256':frozen_hash,'alpha':alpha,'seed':seed,'verified_before_data_generation':True},indent=2)+'\n')
    torch.use_deterministic_algorithms(True)
    study_start=sync_time()
    z=make_initial(32,1000+seed);zc=make_initial(8,2000+seed);zt=make_initial(64,3000+seed)
    teacher_start=sync_time()
    target,train_check=refined_teacher(alpha,z)
    calibration,cal_check=refined_teacher(alpha,zc)
    test,test_check=refined_teacher(alpha,zt)
    teacher_seconds=sync_time()-teacher_start
    scale=target.std(dim=(0,1)).clamp_min(1e-6).detach()
    start=sync_time();initial=params(seed,True)
    common_opt,window_history=warm_start(z,target,scale,initial)
    warm_seconds=sync_time()-start
    common_parameters={k:v.detach().clone() for k,v in initial.items()}
    common_optimizer=copy.deepcopy(common_opt.state_dict())
    def objective(states,p):return ((states-calibration)/scale).square().mean()
    def reference(z,p,h):return solve_observed(acceleration_model,z,p,h,SolverConfig('rk4',512),OBS)
    rotation=(seed-11)%3
    order=list(STRATEGIES[rotation:]+STRATEGIES[:rotation]);results=[]
    for strategy in order:
        policy_start=sync_time()
        p={k:v.clone().requires_grad_(True) for k,v in common_parameters.items()}
        optimizer=torch.optim.Adam(list(p.values()),lr=.003);optimizer.load_state_dict(copy.deepcopy(common_optimizer))
        config=SolverConfig('rk4',64);history=[];selection=None;selection_seconds=0.;snapshot_research_seconds=0.
        s=sync_time();initial_case=ValidationCase('initial',zc,p,H,OBS)
        if strategy=='fixed':snapshot_research_seconds+=sync_time()-s
        snapshots=[initial_case]
        if strategy=='initial_only':
            s=sync_time();selection=evaluate(acceleration_model,objective,[initial_case],CANDIDATES,reference=reference,
                reference_id='RK4/512 independently refined during research diagnostics',workload_id=f'{alpha}-{seed}-{strategy}',repeats=3)
            selection_seconds+=sync_time()-s
            config=SolverConfig(**selection['selected_configuration']) if selection['selected_configuration'] else SolverConfig('rk4',256)
        for update in range(1,201):
            optimizer.zero_grad(set_to_none=True)
            predicted=solve_observed(acceleration_model,z,p,H,config,OBS)
            loss=((predicted-target)/scale).square().mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {strategy}/{update}')
            loss.backward();optimizer.step();history.append(loss.item())
            if update in (20,100,200):
                s=sync_time();case=ValidationCase(f'update{update}',zc,p,H,OBS);snapshots.append(case)
                if not (strategy=='initial_plus_early' and update==20):snapshot_research_seconds+=sync_time()-s
            if strategy=='initial_plus_early' and update==20:
                s=sync_time();selection=evaluate(acceleration_model,objective,snapshots[:2],CANDIDATES,reference=reference,
                    reference_id='RK4/512 independently refined during research diagnostics',workload_id=f'{alpha}-{seed}-{strategy}',repeats=3)
                selection_seconds+=sync_time()-s
                config=SolverConfig(**selection['selected_configuration']) if selection['selected_configuration'] else SolverConfig('rk4',256)
        elapsed=sync_time()-policy_start-snapshot_research_seconds
        # Everything below is separately timed research work; never affects selection.
        diagnostic_start=sync_time();checks=[];reference_checks=[]
        for case in snapshots:
            ref=execute(case,acceleration_model,objective,SolverConfig('rk4',512))
            fine=execute(case,acceleration_model,objective,SolverConfig('rk4',1024))
            tighter=Tolerances(state_atol=1e-7,state_rtol=1e-4,gradient_atol=1e-8,gradient_rtol=1e-3)
            reference_checks.append(compare(case,acceleration_model,objective,SolverConfig('rk4',512),fine,tighter))
            checks.append(compare(case,acceleration_model,objective,config,ref,Tolerances()))
        with torch.no_grad():
            predicted=solve_observed(acceleration_model,zt,p,H,SolverConfig('rk4',512),OBS)
            fine=solve_observed(acceleration_model,zt,p,H,SolverConfig('rk4',1024),OBS)
            test_error=(predicted-fine).norm().item();test_limit=1e-7+1e-4*fine.norm().item()
            test_mse=((predicted-test)/scale).square().mean().item()
        diagnostic_seconds=sync_time()-diagnostic_start
        reference_ok=all(c['passed'] for c in reference_checks) and test_error<=test_limit
        result={'strategy':strategy,'selected_configuration':{'method':config.method,'steps':config.steps},
            'fallback_used':selection is not None and selection['selected_configuration'] is None,
            'policy_seconds':warm_seconds+elapsed,'common_warm_start_seconds_charged':warm_seconds,
            'selection_seconds':selection_seconds,'full_horizon_and_setup_seconds':elapsed-selection_seconds,
            'research_snapshot_seconds':snapshot_research_seconds,'research_diagnostic_seconds':diagnostic_seconds,
            'heldout_test_mse':test_mse,'test_reference_error':test_error,'test_reference_limit':test_limit,
            'references_passed':reference_ok,'reference_checks':reference_checks,'numerical_checks':checks,
            'loss_history':history,'selection':selection}
        results.append(result)
        payload={'alpha':alpha,'seed':seed,'strategy_order':order,'teacher_checks':{'training':train_check,'calibration':cal_check,'heldout_test':test_check},
            'teacher_research_seconds':teacher_seconds,'window_loss_history':window_history,'scale':scale.cpu().tolist(),'results':results}
        (output/'policies.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
        torch.save({'snapshots':[c.manifest() for c in snapshots],'parameters':{k:v.detach().cpu() for k,v in p.items()},
            'optimizer':optimizer.state_dict(),'train_initial':z.cpu(),'calibration_initial':zc.cpu(),'heldout_test_initial':zt.cpu(),
            'teacher':target.cpu(),'calibration_teacher':calibration.cpu(),'heldout_test_teacher':test.cpu()},output/f'{strategy}.pt')
        print(strategy,config,'cost',result['policy_seconds'],'test_mse',test_mse,'references',reference_ok,
            'late',[(c['case'],c['passed']) for c in checks[2:]],flush=True)
    fixed=next(r for r in results if r['strategy']=='fixed')
    for r in results:
        r['quality_allowance']=fixed['heldout_test_mse']*1.05+1e-4
        r['useful_quality_cost']=r['strategy']!='fixed' and r['references_passed'] and fixed['references_passed'] and r['policy_seconds']<fixed['policy_seconds'] and r['heldout_test_mse']<=r['quality_allowance']
    payload['study_seconds']=sync_time()-study_start
    (output/'policies.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    files=[Path(__file__),Path('examples/development_pilot.py'),Path('src/dynamicsbench/core.py'),Path('research/FINAL_PROTOCOL.md')]
    (output/'manifest.json').write_text(json.dumps({'development_only':False,'reserved_seed_evaluation':True,'freeze_sha256':frozen_hash,'alpha':alpha,'seed':seed,
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'float64','model':'known_kinematics_learned_acceleration',
        'common_window_updates':100,'full_horizon_updates':200,'dataset_seeds':{'train':1000+seed,'calibration':2000+seed,'heldout_test':3000+seed},
        'paired_parameters_and_optimizer':True,'hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--alpha',type=int,choices=(1,16),required=True)
    parser.add_argument('--seed',type=int,choices=(11,12,13),required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,default=Path('research/final-freeze.json'))
    args=parser.parse_args()
    try:run(args.alpha,args.seed,args.output,args.freeze)
    except Exception as error:
        if args.output.exists() and not (args.output/'failure.json').exists():
            (args.output/'failure.json').write_text(json.dumps({'error':type(error).__name__,'message':str(error),'alpha':args.alpha,'seed':args.seed},indent=2)+'\n')
        raise
