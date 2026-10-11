import json,tempfile,unittest
from pathlib import Path
from dynamicsbench import load_workload,run_workload
from dynamicsbench.workflow import source_record
ROOT=Path(__file__).resolve().parents[1]
class PortableSourceChecks(unittest.TestCase):
    def test_line_endings_and_real_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.py';p.write_bytes(b'x=1\ny=2\n');a=source_record(p,tmp)
            p.write_bytes(b'x=1\r\ny=2\r\n');b=source_record(p,tmp)
            self.assertNotEqual(a['sha256'],b['sha256']);self.assertEqual(a['canonical_sha256'],b['canonical_sha256'])
            p.write_bytes(b'x=9\n');self.assertNotEqual(a['canonical_sha256'],source_record(p,tmp)['canonical_sha256'])
    def test_default_exports_and_external_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=load_workload(ROOT/'examples/product_linear.py');run_workload(w,Path(tmp)/'run',repeats=3)
            for name in ('report.json','recommendation.json','provenance.json'):
                text=(Path(tmp)/'run'/name).read_text();self.assertNotIn(str(ROOT),text)
                data=json.loads(text);sources=data.get('source_files',data.get('product',{}).get('source_files'))
                self.assertEqual(sources[0]['path'],'examples/product_linear.py')
            p=Path(tmp)/'private_username.py';p.write_text('x=1')
            record=source_record(p,ROOT);self.assertTrue(record['path'].startswith('external/source-'))
            self.assertNotIn('private_username',record['path'])
            self.assertEqual(source_record(p,ROOT,'absolute')['path'],str(p.resolve()))
    def test_absolute_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=load_workload(ROOT/'examples/product_linear.py');r=run_workload(w,tmp,repeats=3,source_paths='absolute')
            self.assertEqual(r['product']['source_path_mode'],'absolute')
            self.assertEqual(r['product']['source_files'][0]['path'],str((ROOT/'examples/product_linear.py').resolve()))
