import unittest
import torch
from dynamicsbench.core import ValidationCase, SolverConfig, Tolerances, metrics, execute, evaluate

def dyn(t,z,p): return p['rate']*z

def obj(z,p): return .5*z.square().sum()

def ref(z,p,h): return z*torch.exp(p['rate']*h)

def case(name='one',rate=-1.):
    return ValidationCase(name,torch.ones(2,dtype=torch.float64),{'rate':torch.full((2,),rate,dtype=torch.float64)})

def evaluate_one(candidates,heldout=()):
    return evaluate(dyn,obj,[case()],candidates,reference=ref,reference_id='exact',workload_id='test',heldout_cases=heldout,repeats=3)

class CoreChecks(unittest.TestCase):
    def test_vector_analytical_gradient(self):
        state,grads=execute(case(),dyn,obj,SolverConfig('rk4',128))
        self.assertTrue(torch.allclose(grads['rate'],torch.full((2,),torch.exp(torch.tensor(-2.)).item(),dtype=torch.float64),atol=1e-8,rtol=0))
    def test_snapshot_independence(self):
        p=torch.tensor([-1.,-2.],dtype=torch.float64)
        c=ValidationCase('snapshot',torch.ones(2,dtype=torch.float64),{'rate':p})
        before=c.manifest();p.fill_(5)
        z,q=c.fresh();q['rate'].data.fill_(9);z.fill_(0)
        self.assertEqual(before,c.manifest())
    def test_heldout_does_not_change_selection(self):
        config=SolverConfig('rk4',8)
        r=evaluate_one([config],[case('late',-20.)])
        self.assertEqual(r['selected_configuration'],{'method':'rk4','steps':8})
        self.assertFalse(r['heldout_passed'])
    def test_no_passing_candidate(self):
        self.assertIsNone(evaluate_one([SolverConfig('euler',1)])['selected_configuration'])
    def test_parameter_groups_independent(self):
        z=torch.ones(2,dtype=torch.float64)
        c=ValidationCase('groups',z,{'used':-z,'unused':z})
        _,g=execute(c,lambda t,z,p:p['used']*z,obj,SolverConfig('rk4',16))
        self.assertTrue(torch.equal(g['unused'],torch.zeros_like(z)))
    def test_zero_norm_and_nonfinite(self):
        z=torch.zeros(2,dtype=torch.float64)
        m=metrics(z,z,1e-7,1e-2,1e-10)
        self.assertTrue(m['passed']);self.assertIsNone(m['cosine']);self.assertIsNone(m['relative_l2'])
        self.assertFalse(metrics(z+float('nan'),z,1e-7,1e-2,1e-10)['passed'])
    def test_vector_direction_and_threshold(self):
        b=torch.tensor([1.,-1.],dtype=torch.float64)
        self.assertAlmostEqual(metrics(-b,b,0,0,1e-10)['cosine'],-1)
        self.assertFalse(metrics(-b,b,0,0,1e-10)['passed'])
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):SolverConfig('rk4',0)
        with self.assertRaises(ValueError):Tolerances(gradient_atol=-1)
        with self.assertRaises(ValueError):evaluate_one([SolverConfig('rk4',16)],[case()])
    def test_invalid_reference(self):
        with self.assertRaises(ValueError):
            evaluate(dyn,obj,[case()],[SolverConfig('rk4',16)],reference=lambda z,p,h:z*float('nan'),reference_id='bad',workload_id='test',repeats=3)
    def test_objective_exception_becomes_failure(self):
        def bad(z,p):
            if z.max()>.37: raise ValueError('unsupported state')
            return obj(z,p)
        r=evaluate(dyn,bad,[case()],[SolverConfig('rk4',1)],reference=ref,reference_id='exact',workload_id='test',repeats=3)
        self.assertIsNone(r['selected_configuration'])
        self.assertIn('unsupported state',r['results'][0]['checks'][0]['failure'])

if __name__=='__main__':unittest.main()
