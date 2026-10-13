import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
from policy_evaluation import verify_freeze

class EvaluationFreezeTests(unittest.TestCase):
    def test_integrity_and_reserved_seed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'input.py';file.write_text('original')
            freeze=root/'freeze.json';freeze.write_text(json.dumps({'evaluation_seeds':[11,12,13],'alphas':[1,16],
                'files':{'input.py':hashlib.sha256(file.read_bytes()).hexdigest()}}))
            self.assertEqual(verify_freeze(freeze,root,1,11),hashlib.sha256(freeze.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError,'Reserved'):verify_freeze(freeze,root,1,0)
            file.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen input changed'):verify_freeze(freeze,root,1,12)
