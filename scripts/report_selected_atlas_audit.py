"""Read-only E106 same-selected-checkpoint audit report."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args()
    run=a.run or max((p for p in Path('/home/carsonc/r-elegans-results').glob('selected-atlas-audit-*') if p.is_dir()),default=None)
    if run is None:raise SystemExit('No selected-checkpoint audit found')
    print('Run:',run)
    path=run/'summary.json'
    if not path.exists():print('No complete cohort summary; inspect adjacent .log and cases/*/summary.json')
    else:
        s=json.loads(path.read_text());print('Audit complete:',s['audit_complete'],'Preparation failures:',s['preparation_failures'])
        for r in s['finished']:
            result=r['result'];checks=result.get('checks',{});numerical=checks.get('selected_substep_convergence',{});rhythm=checks.get('rhythm',{})
            print(r['label'],dict(complete=result['audit_complete'],checkpoint_sha256=result.get('checkpoint_sha256'),
                integrity_and_atlas_numerics=result.get('integrity_and_atlas_numerics_passed'),problems=result.get('problems'),
                selected_relative_numerical_error=numerical.get('max_abs_difference_over_max_abs_response'),
                selected_numerical_pass=numerical.get('passed'),rhythm_component_pass=rhythm.get('component_criteria_passed'),
                strict_rhythm_convergence=rhythm.get('strict_convergence_passed'),validation_leakage=result.get('validation_leakage'),
                selected_peptide_tau_s=checks.get('peptides',{}).get('tau_s'),
                selection_baseline_fresh=checks.get('selection',{}).get('baseline_resettled_at_selected_step')))
            for name,metric in result.get('selected_metrics',{}).items():print(' ',name,{k:metric.get(k) for k in ('spearman_mean_z','auc_excited','model_silent_fraction_among_significant')})
            if r.get('declared_performance'):print(' Original performance gates:',r['declared_performance'])
            if result.get('selected_peptide_off_metrics'):print(' peptide-off test:',result['selected_peptide_off_metrics']['test'])
    if (run/'budget_receipt.json').exists():print('Budget:',json.loads((run/'budget_receipt.json').read_text()))
    print('No model promoted; rhythm and atlas checks do not establish body gait, sensory start/stop or swimming.')


if __name__=='__main__':main()
