"""Descriptive frozen-evaluation reporting; no decision rules are fitted here."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def summarize(root):
    files=sorted(root.glob('alpha*-seed*/policies.json'))
    runs=[json.loads(p.read_text()) for p in files]
    expected={(a,s) for a in (1,16) for s in (11,12,13)}
    if {(r['alpha'],r['seed']) for r in runs}!=expected or any(len(r['results'])!=3 or 'study_seconds' not in r for r in runs):
        raise ValueError('Need all six complete paired evaluation cases; report failures explicitly rather than silently dropping cases')
    rows=[];aggregates={};fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    colors={'fixed':'#777777','initial_only':'#2878b5','initial_plus_early':'#d35c24'}
    for run in runs:
        fixed=next(r for r in run['results'] if r['strategy']=='fixed')
        ax=axes[0 if run['alpha']==1 else 1,run['seed']-11]
        for r in run['results']:
            checks=[c for c in r['numerical_checks'] if c['case'] in ('update100','update200')]
            failed=[c['case'] for c in checks if not c['passed']]
            row={'alpha':run['alpha'],'seed':run['seed'],'strategy':r['strategy'],
                'solver':f"{r['selected_configuration']['method']}/{r['selected_configuration']['steps']}",
                'policy_seconds':r['policy_seconds'],'selection_seconds':r['selection_seconds'],
                'warm_start_seconds':r['common_warm_start_seconds_charged'],'heldout_test_mse':r['heldout_test_mse'],
                'cost_ratio_to_fixed':r['policy_seconds']/fixed['policy_seconds'],
                'mse_ratio_to_fixed':r['heldout_test_mse']/fixed['heldout_test_mse'],
                'quality_allowance':r['quality_allowance'],'references_passed':r['references_passed'],
                'late_pass':not failed,'failed_late_checkpoints':','.join(failed),'useful_quality_cost':r['useful_quality_cost'],
                'fallback_used':r['fallback_used'],'early_pass_late_failure':r['strategy']!='fixed' and not r['fallback_used'] and bool(failed),
                'research_diagnostic_seconds':r['research_diagnostic_seconds']}
            rows.append(row)
            key=f"alpha{run['alpha']}/{r['strategy']}"
            a=aggregates.setdefault(key,{'cases':0,'useful_quality_cost_count':0,'late_pass_count':0,'reference_pass_count':0,
                'fallback_count':0,'early_pass_late_failure_count':0,'cost_ratios':[],'mse_ratios':[]})
            a['cases']+=1;a['useful_quality_cost_count']+=bool(row['useful_quality_cost']);a['late_pass_count']+=row['late_pass']
            a['reference_pass_count']+=row['references_passed'];a['fallback_count']+=row['fallback_used']
            a['early_pass_late_failure_count']+=row['early_pass_late_failure'];a['cost_ratios'].append(row['cost_ratio_to_fixed']);a['mse_ratios'].append(row['mse_ratio_to_fixed'])
            ax.scatter(row['policy_seconds'],row['heldout_test_mse'],color=colors[r['strategy']],s=50,label=r['strategy'])
        ax.axhline(fixed['quality_allowance'],color='#999999',linestyle=':',label='quality allowance')
        ax.set_title(f"alpha={run['alpha']}, seed={run['seed']}");ax.set_xlabel('Charged policy cost (s)')
        ax.set_ylabel('Held-out normalized MSE');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Frozen reserved-seed evaluation: quality versus total policy cost')
    fig.savefig(root/'quality-cost.png',dpi=180);fig.savefig(root/'quality-cost.pdf')
    with (root/'evaluation-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'paired_cases':6,'strategy_runs':18,'aggregates':aggregates,'actual_case_execution_seconds':sum(r['study_seconds'] for r in runs),
        'note':'Three paired seeds per regime; descriptive ratios/counts only. Summed case wall times are not CPU/GPU utilization or whole-study wall time.'}
    (root/'evaluation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return rows,summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);summarize(parser.parse_args().root)
