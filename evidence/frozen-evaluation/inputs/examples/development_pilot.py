"""Development-only oscillator learning; no evaluation seeds or policy comparison."""
import argparse
import json
import time
from pathlib import Path
import torch
from dynamicsbench.core import solve_observed,SolverConfig,ValidationCase,evaluate,execute,metrics

OBS=16;H=6.

def teacher(alpha,z,steps):
    def dyn(t,z,p):
        q,v=z.unbind(-1)
        return torch.stack((v,-alpha*q-q**3-.15*v),-1)
    return solve_observed(dyn,z,{},H,SolverConfig('rk4',steps),OBS)

def make_initial(n,seed):
    g=torch.Generator().manual_seed(seed)
    return (torch.rand(n,2,generator=g,dtype=torch.float64)*2-1).cuda()

def params(seed, acceleration_only=False):
    g=torch.Generator().manual_seed(seed)
    shapes={'w1':(2,32),'w2':(32,32),'w3':(32,1 if acceleration_only else 2)}
    p={k:(torch.randn(shape,generator=g,dtype=torch.float64)*(.3 if k=='w3' else .2)).cuda().requires_grad_() for k,shape in shapes.items()}
    p.update({k:torch.zeros(n,device='cuda',dtype=torch.float64,requires_grad=True) for k,n in [('b1',32),('b2',32),('b3',1 if acceleration_only else 2)]})
    return p

def model(t,z,p):return torch.tanh(torch.tanh(z@p['w1']+p['b1'])@p['w2']+p['b2'])@p['w3']+p['b3']

def acceleration_model(t,z,p):
    acceleration=model(t,z,p)
    return torch.cat((z[...,1:2],acceleration),dim=-1)

def run(output,epochs,acceleration_only=False,window_updates=0):
    dynamics=acceleration_model if acceleration_only else model
    output.mkdir(parents=True,exist_ok=True)
    torch.use_deterministic_algorithms(True)
    results=[]
    for alpha in (1.,16.):
        for seed in (0,1):
            z=make_initial(32,1000+seed);zc=make_initial(8,2000+seed)
            with torch.no_grad():
                target=teacher(alpha,z,1024);refined=teacher(alpha,z,2048)
                tc=teacher(alpha,zc,1024);tc_refined=teacher(alpha,zc,2048)
            terr=(target-refined).norm().item();tn=refined.norm().item()
            if terr>1e-8+1e-5*tn:raise RuntimeError('Teacher refinement failed')
            cerr=(tc-tc_refined).norm().item();cn=tc_refined.norm().item()
            if cerr>1e-8+1e-5*cn:raise RuntimeError('Calibration teacher refinement failed')
            scale=target.std(dim=(0,1)).clamp_min(1e-6).detach()
            p=params(seed,acceleration_only)
            def obj(states,ps):return ((states-tc)/scale).square().mean()
            snapshots=[];history=[];window_history=[]
            opt=torch.optim.Adam(list(p.values()),lr=.003)
            torch.cuda.synchronize();start=time.perf_counter()
            if window_updates:
                # Autonomous dynamics: observed states provide phase-diverse starts.
                starts=torch.cat((z.unsqueeze(0),target[:-1]),dim=0).reshape(-1,2).detach()
                ends=target.reshape(-1,2).detach()
                for _ in range(window_updates):
                    opt.zero_grad(set_to_none=True)
                    predicted=solve_observed(dynamics,starts,p,H/OBS,SolverConfig('rk4',8),1)
                    window_loss=((predicted-ends)/scale).square().mean()
                    if not torch.isfinite(window_loss):raise RuntimeError('Nonfinite window loss')
                    window_loss.backward();opt.step();window_history.append(window_loss.item())
            torch.cuda.synchronize();window_seconds=time.perf_counter()-start
            snapshots.append(ValidationCase('initial',zc,p,H,OBS))
            torch.cuda.synchronize();start=time.perf_counter()
            for step in range(epochs):
                opt.zero_grad(set_to_none=True)
                states=solve_observed(dynamics,z,p,H,SolverConfig('rk4',64),OBS)
                loss=((states-target)/scale).square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite development loss')
                loss.backward();opt.step();history.append(loss.item())
                if step in (19,99,epochs-1):snapshots.append(ValidationCase(f'update{step+1}',zc,p,H,OBS))
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            def reference(initial,ps,h):return solve_observed(dynamics,initial,ps,h,SolverConfig('rk4',256),OBS)
            refinement=[]
            for case in snapshots:
                a=execute(case,dynamics,obj,SolverConfig('rk4',256));b=execute(case,dynamics,obj,SolverConfig('rk4',512))
                sm=[metrics(x,y,1e-7,1e-4,1e-10) for x,y in zip(a[0],b[0])]
                gm={k:metrics(v,b[1][k],1e-8,1e-3,1e-10) for k,v in a[1].items()}
                refinement.append({'case':case.name,'passed':all(m['passed'] for m in sm+list(gm.values())),'states':sm,'gradients':gm})
            report=evaluate(dynamics,obj,snapshots[:2],[SolverConfig('euler',32),SolverConfig('euler',64),SolverConfig('rk4',16),SolverConfig('rk4',32),SolverConfig('rk4',64)],
                reference=reference,reference_id='RK4/256 refined against512',workload_id=f'dev-alpha{alpha}-seed{seed}',heldout_cases=snapshots[2:],repeats=3)
            with torch.no_grad():
                pred=solve_observed(dynamics,zc,p,H,SolverConfig('rk4',256),OBS)
                cal_mse=((pred-tc)/scale).square().mean().item()
                initial_model=snapshots[0].fresh()[1]
                initpred=solve_observed(dynamics,zc,initial_model,H,SolverConfig('rk4',256),OBS)
                initial_mse=((initpred-tc)/scale).square().mean().item()
            row={'alpha':alpha,'seed':seed,'teacher_refinement':{'absolute_l2':terr,'reference_l2':tn},'calibration_teacher_refinement':{'absolute_l2':cerr,'reference_l2':cn},'architecture':'known_kinematics_learned_acceleration' if acceleration_only else 'generic_vector_field','scale':scale.cpu().tolist(),
                'training_seconds':elapsed,'window_training_seconds':window_seconds,'window_updates':window_updates,'window_loss_history':window_history,'loss_history':history,'initial_calibration_mse':initial_mse,'final_calibration_mse':cal_mse,
                'model_refinement':refinement,'selection':report}
            results.append(row)
            (output/'pilot.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
            torch.save({'parameters':{k:v.detach().cpu() for k,v in p.items()},'train_initial':z.cpu(),'calibration_initial':zc.cpu(),'teacher':target.cpu(),'calibration_teacher':tc.cpu()},output/f'alpha{int(alpha)}-seed{seed}.pt')
            print(alpha,seed,'mse',initial_mse,cal_mse,'reference_pass',all(x['passed'] for x in refinement),'selected',report['selected_configuration'],'late',report['heldout_passed'],flush=True)
    (output/'manifest.json').write_text(json.dumps({'seeds':[0,1],'alphas':[1,16],'epochs':epochs,'window_updates':window_updates,'selection_initial_is_post_window_training':bool(window_updates),'architecture':'known_kinematics_learned_acceleration' if acceleration_only else 'generic_vector_field','dtype':'float64','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'development_only':True,'no_heldout_evaluation_seeds':True},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/pace/development'));p.add_argument('--epochs',type=int,default=200);p.add_argument('--acceleration-only',action='store_true');p.add_argument('--window-updates',type=int,default=0);a=p.parse_args();run(a.output,a.epochs,a.acceleration_only,a.window_updates)
