import unittest
import torch
from dynamicsbench.core import ValidationCase,SolverConfig,execute,metrics
from dynamicsbench.execution import Runner

def dyn(t,z,p):return p['rates']*z

def obj(z,p):return z.square().mean()/2

def ref(z,p,h):return z*torch.exp(p['rates']*h)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class CudaChecks(unittest.TestCase):
    def test_eager_and_compiled_states_and_gradients(self):
        for dtype in (torch.float32,torch.float64):
            case=ValidationCase('cuda',torch.ones(4,8,device='cuda',dtype=dtype),{'rates':torch.linspace(-.3,-.1,8,device='cuda',dtype=dtype)})
            config=SolverConfig('rk4',8)
            reference=execute(case,dyn,obj,config,ref)
            for backend in ('eager','compiled'):
                state,grads=Runner(dyn,obj,config,backend)(case)
                self.assertTrue(metrics(state,reference[0],1e-6,1e-3,1e-10)['passed'])
                self.assertTrue(metrics(grads['rates'],reference[1]['rates'],1e-7,1e-2,1e-10)['passed'])
    def test_compiled_trajectory_objective(self):
        case=ValidationCase('observed',torch.ones(2,4,device='cuda',dtype=torch.float64),{'rates':torch.linspace(-.3,-.1,4,device='cuda',dtype=torch.float64)},observations=4)
        reference=lambda z,p,h:torch.stack([z*torch.exp(p['rates']*h*i/4) for i in range(1,5)])
        expected=execute(case,dyn,obj,SolverConfig('rk4',16),reference)
        actual=Runner(dyn,obj,SolverConfig('rk4',16),'compiled')(case)
        self.assertTrue(metrics(actual[0],expected[0],1e-6,1e-3,1e-10)['passed'])
        self.assertTrue(metrics(actual[1]['rates'],expected[1]['rates'],1e-7,1e-2,1e-10)['passed'])
    def test_mixed_device_rejected(self):
        with self.assertRaises(ValueError):ValidationCase('mixed',torch.ones(2,device='cuda'),{'p':torch.ones(2)})
