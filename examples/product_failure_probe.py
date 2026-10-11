"""Deliberately failing candidate to demonstrate retained no-selection exports."""
import torch
from dynamicsbench import Workload,ValidationCase,SolverConfig

def build_workload(device='cpu'):
    z=torch.ones(1,dtype=torch.float64,device=device)
    return Workload(lambda t,z,p:p['rate']*z,lambda z,p:.5*z.square().sum(),
        lambda z,p,h:z*torch.exp(p['rate']*h),[ValidationCase('initial',z,{'rate':-z})],
        [SolverConfig('euler',1)],'failure-probe','analytical exponential')
