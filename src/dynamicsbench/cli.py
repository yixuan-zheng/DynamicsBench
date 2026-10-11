"""Installed CLI using the same evaluation/export path as the Python API."""
import argparse
from pathlib import Path
from .workflow import load_workload, run_workload

def main():
    parser=argparse.ArgumentParser(description='Validate and rank solver settings for a trusted Python workload.')
    parser.add_argument('--workload',type=Path,required=True,help='Python file defining build_workload(device=...)')
    parser.add_argument('--factory',default='build_workload')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True,help='New or empty artifact directory')
    parser.add_argument('--absolute-source-paths',action='store_true',help='Opt in to absolute source paths for private local debugging')
    args=parser.parse_args()
    try:
        workload=load_workload(args.workload,device=args.device,factory=args.factory)
        report=run_workload(workload,args.output,repeats=args.repeats,source_paths="absolute" if args.absolute_source_paths else "relative")
    except (ValueError,RuntimeError,ImportError,OSError) as exc:
        parser.exit(2,f'DynamicsBench: {exc}\n')
    print(f"Status: {report['product']['status']}; selected: {report['selected_configuration']}")
    print(f'Report and recommendation: {args.output.resolve()}')
    if report['product']['status']!='selected':raise SystemExit(3)

if __name__=='__main__':main()
