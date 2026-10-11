import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import torch
from dynamicsbench import Workload, ValidationCase, SolverConfig, Tolerances, run_workload, load_workload
from dynamicsbench.core import execute

ROOT=Path(__file__).resolve().parents[1]
class ProductChecks(unittest.TestCase):
    def workload(self):return load_workload(ROOT/'examples/product_linear.py')
    def test_api_cli_validation_parity_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=run_workload(self.workload(),Path(tmp)/'api',repeats=3)
            result=subprocess.run([sys.executable,'-m','dynamicsbench.cli','--workload',str(ROOT/'examples/product_linear.py'),'--output',str(Path(tmp)/'cli'),'--repeats','3'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            b=json.loads((Path(tmp)/'cli/report.json').read_text())
            self.assertEqual(a['manifest']['cases'],b['manifest']['cases'])
            self.assertEqual([(r['configuration'],r['checks']) for r in a['results']],[(r['configuration'],r['checks']) for r in b['results']])
            for name in ('rankings.csv','recommendation.json','provenance.json','environment-lock.txt'):
                self.assertTrue((Path(tmp)/'cli'/name).is_file())
            with self.assertRaisesRegex(ValueError,'fresh run'):run_workload(self.workload(),Path(tmp)/'cli')
    def test_no_selection_and_heldout_failure_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=self.workload();w.candidates=[SolverConfig('euler',1)]
            r=run_workload(w,Path(tmp)/'none',repeats=3)
            self.assertEqual(r['product']['status'],'no_recommendation')
            self.assertIsNone(json.loads((Path(tmp)/'none/recommendation.json').read_text())['configuration'])
            w=self.workload();w.candidates=[SolverConfig('rk4',8)]
            z=torch.ones(2,dtype=torch.float64)
            w.heldout_cases=[ValidationCase('hard',z,{'rate':-20*z})]
            r=run_workload(w,Path(tmp)/'heldout',repeats=3)
            self.assertEqual(r['product']['status'],'heldout_failed')
            self.assertFalse(json.loads((Path(tmp)/'heldout/recommendation.json').read_text())['usable_on_tested_cases'])
    def test_boundaries(self):
        for n in (1.5,True,0):
            with self.assertRaises(ValueError):SolverConfig('rk4',n)
        for h in (float('inf'),float('nan')):
            with self.assertRaises(ValueError):ValidationCase('bad',torch.ones(1),{'x':torch.ones(1)},h)
        with self.assertRaises(ValueError):Tolerances(state_rtol=float('nan'))
        w=self.workload()
        with self.assertRaisesRegex(ValueError,'scalar tensor'):execute(w.selection_cases[0],w.dynamics,lambda z,p:1.,w.candidates[0])
        with tempfile.TemporaryDirectory() as tmp:
            w.reference=lambda z,p,h:z*float('nan')
            with self.assertRaisesRegex(ValueError,'reference failed'):run_workload(w,tmp,repeats=3)
    def test_invalid_grid_is_reported(self):
        w=self.workload();z=torch.ones(2,dtype=torch.float64)
        w.selection_cases=[ValidationCase('grid',z,{'rate':-z},observations=3)];w.heldout_cases=[]
        w.reference=lambda z,p,h:torch.stack([z*torch.exp(p['rate']*h*i/3) for i in (1,2,3)])
        w.candidates=[SolverConfig('rk4',4)]
        with tempfile.TemporaryDirectory() as tmp:
            r=run_workload(w,tmp,repeats=3)
            self.assertEqual(r['product']['status'],'no_recommendation')
            self.assertIn('divisible',r['results'][0]['checks'][0]['failure'])
    def test_missing_factory_and_cuda_capability(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'empty.py';p.write_text('')
            with self.assertRaisesRegex(ValueError,'build_workload'):load_workload(p)
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError,'CUDA unavailable'):load_workload(ROOT/'examples/product_linear.py',device='cuda')

if __name__=='__main__':unittest.main()
