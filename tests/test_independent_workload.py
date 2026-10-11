"""Integration contracts for the pinned external PyTorch workload."""
import importlib.util
from pathlib import Path
import unittest
import torch

@unittest.skipUnless(importlib.util.find_spec('torchdiffeq'), 'optional P1 dependency')
class IndependentWorkloadChecks(unittest.TestCase):
    def test_functional_parameters_and_objective_equivalence(self):
        path=Path(__file__).resolve().parents[1]/'examples/independent_workload.py'
        spec=importlib.util.spec_from_file_location('independent_workload',path)
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        w=adapter.make_workload()
        z=w['initial'];t=z.new_tensor(0.)
        before={k:p.detach().clone() for k,p in w['module'].named_parameters()}
        p={k:v.clone().requires_grad_() for k,v in w['parameters'].items()}
        actual=w['dynamics'](t,z,p)
        self.assertTrue(torch.equal(actual,w['module'](t,z)))
        grads=torch.autograd.grad(actual.square().sum(),tuple(p.values()))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(g.abs().sum()>0 for g in grads))
        changed={k:v.clone() for k,v in p.items()};changed['net.2.bias']=changed['net.2.bias']+.5
        self.assertTrue(torch.allclose(w['dynamics'](t,z,changed),actual+.5))
        states=w['target'][1:]+.01
        expected=torch.mean(torch.abs(torch.cat((z.unsqueeze(0),states))-w['target']))
        self.assertTrue(torch.equal(w['objective'](states,p),expected))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in w['module'].named_parameters()))

if __name__=='__main__':unittest.main()
