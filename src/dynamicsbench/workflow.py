"""Shared user-workload evaluation and export path for Python and CLI."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import time
import torch
from .core import evaluate, Tolerances

@dataclass
class Workload:
    dynamics: object
    objective: object
    reference: object
    selection_cases: list
    candidates: list
    workload_id: str
    reference_id: str
    heldout_cases: list = field(default_factory=list)
    tolerances: Tolerances = field(default_factory=Tolerances)
    source_files: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source_root: object = None

def load_workload(path, *, device='cpu', factory='build_workload'):
    """Load trusted Python code; factory(device=...) must return Workload."""
    path=Path(path).resolve()
    if not path.is_file():raise ValueError(f'Workload file does not exist: {path}')
    _device(device)
    spec=importlib.util.spec_from_file_location('dynamicsbench_user_'+hashlib.sha256(str(path).encode()).hexdigest()[:12],path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build=getattr(module,factory,None)
    if not callable(build):raise ValueError(f'Workload must define {factory}(device=...)')
    workload=build(device=device)
    if not isinstance(workload,Workload):raise ValueError('Factory must return dynamicsbench.Workload')
    workload.source_files=list(dict.fromkeys([path,*map(Path,workload.source_files)]))
    if workload.source_root is None:
        workload.source_root=next((parent for parent in path.parents if (parent/'pyproject.toml').is_file()),path.parent)
    return workload

def _device(device):
    d=torch.device(device)
    if d.type not in ('cpu','cuda'):raise ValueError('Supported devices: cpu or cuda')
    if d.type=='cuda':
        if not torch.cuda.is_available():raise ValueError('CUDA unavailable: use --device cpu or install a CUDA-enabled PyTorch on an allocated GPU node')
        if d.index is not None and d.index>=torch.cuda.device_count():raise ValueError('Requested CUDA device index is unavailable')
        try:
            torch.ones(1,device=d).add_(1)
            torch.cuda.synchronize(d)
        except RuntimeError as exc:
            raise ValueError(f'CUDA preflight failed for {torch.cuda.get_device_name(d)} with PyTorch {torch.__version__}: {exc}. Use a compatible PyTorch build/GPU profile or CPU.') from exc
    return d

def source_record(path, root, mode='relative'):
    """Raw hashes identify bytes; canonical text hashes normalize CRLF only."""
    p=Path(path).resolve();data=p.read_bytes()
    raw=hashlib.sha256(data).hexdigest()
    text=p.suffix.lower() in ('.py','.md','.json','.toml','.txt','.csv') or p.name in ('.gitattributes','.gitignore','LICENSE')
    if text:
        try:data.decode('utf-8')
        except UnicodeDecodeError:text=False
    canonical=data.replace(b'\r\n',b'\n') if text else data
    if mode=='absolute':label=str(p)
    else:
        try:label=p.relative_to(Path(root).resolve()).as_posix()
        except ValueError:label='external/source-'+raw[:16]
    return {'path':label,'sha256':raw,'canonical_sha256':hashlib.sha256(canonical).hexdigest(),
        'canonicalization':'utf8-crlf-to-lf' if text else 'raw-bytes'}

def run_workload(workload, output, *, repeats=7, source_paths='relative'):
    """Write a fresh artifact directory. Nonempty output is never overwritten."""
    if not isinstance(workload,Workload):raise ValueError('Expected Workload')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be a new or empty directory; choose a fresh run name')
    if not workload.workload_id or not workload.reference_id:raise ValueError('Require workload_id and reference_id')
    for callback in (workload.dynamics,workload.objective,workload.reference):
        if not callable(callback):raise ValueError('Dynamics, objective and reference must be callable')
    for case in [*workload.selection_cases,*workload.heldout_cases]:_device(case._initial.device)
    if source_paths not in ('relative','absolute'):raise ValueError('source_paths must be relative or absolute')
    root=workload.source_root or Path.cwd()
    sources=[]
    for source in workload.source_files:
        p=Path(source).resolve()
        if not p.is_file():raise ValueError(f'Source file missing: {p}')
        sources.append(source_record(p,root,source_paths))
    dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    # Check JSON metadata before running expensive evaluation.
    json.dumps(workload.metadata,allow_nan=False)
    start=time.perf_counter()
    try:
        report=evaluate(workload.dynamics,workload.objective,workload.selection_cases,workload.candidates,
            reference=workload.reference,reference_id=workload.reference_id,workload_id=workload.workload_id,
            heldout_cases=workload.heldout_cases,tolerances=workload.tolerances,repeats=repeats)
    except (ValueError,RuntimeError) as exc:
        raise ValueError(f'Evaluation/reference failed: {exc}. Check callback shapes, objective and differentiable reference.') from exc
    elapsed=(time.perf_counter()-start)*1000
    passing=sorted((r for r in report['results'] if r['passed']),key=lambda r:r['timing_ms']['median'])
    ranking=[{'rank':i+1,'configuration':row['configuration'],'timing_ms':row['timing_ms']} for i,row in enumerate(passing)]
    status='no_recommendation' if not report['selected_configuration'] else ('heldout_failed' if report['heldout_passed'] is False else 'selected')
    report['product']={'version':importlib.metadata.version('dynamicsbench'),'status':status,'eligible_ranking':ranking,
        'source_files':sources,'source_path_mode':source_paths,'metadata':workload.metadata,'workflow_elapsed_ms':elapsed,
        'workflow_timing_scope':'Evaluation including reference, checks, warmups and candidate timing; excludes factory setup and export',
        'source_capture_scope':'Only explicitly supplied files plus loader file; imports, closures, datasets and reference validity are not automatically certified.'}
    recommendation={'schema_version':1,'status':status,'configuration':report['selected_configuration'],
        'usable_on_tested_cases':status=='selected','heldout_passed':report['heldout_passed'],
        'workload_id':workload.workload_id,'reference_id':workload.reference_id,
        'cases':report['manifest']['cases'],'tolerances':asdict(workload.tolerances),
        'environment':{k:v for k,v in report['manifest'].items() if k not in ('cases',)},
        'source_files':sources,'source_path_mode':source_paths,'timing_overlap_candidates':report['timing_overlap_candidates'],
        'scope':report['scope'],'report_file':'report.json'}
    output.mkdir(parents=True,exist_ok=True)
    for name,data in [('report.json',report),('recommendation.json',recommendation),('provenance.json',{'source_files':sources,'source_path_mode':source_paths,'dependencies':dependencies,'metadata':workload.metadata})]:
        (output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (output/'environment-lock.txt').write_text(''.join(f'{k}=={v}\n' for k,v in sorted(dependencies.items(),key=lambda x:x[0].lower())))
    with (output/'rankings.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['rank','method','steps','median_ms','min_ms','max_ms'])
        for row in ranking:
            writer.writerow([row['rank'],row['configuration']['method'],row['configuration']['steps'],*[row['timing_ms'][k] for k in ('median','min','max')]])
    return report
