"""Read-only E106 same-selected-checkpoint audit report."""
import argparse,json
from pathlib import Path


def print_details(run, checks):
    """Expose saved evidence without importing the simulator or changing gates."""
    for key in ('parameters', 'selection', 'selected_substep_convergence',
                'peptide_off_substep_convergence'):
        print(' Detail', key + ':', json.dumps(checks.get(key, 'unavailable'), sort_keys=True))
    rhythm = checks.get('rhythm', {})
    reference_path = run / 'references' / str(rhythm.get('reference_checkpoint_sha256', 'missing')) / 'summary.json'
    reference = {}
    try:
        reference = json.loads(reference_path.read_text())
    except (OSError, ValueError) as exc:
        print(' Rhythm reference unavailable:', reference_path, str(exc))
    expected = rhythm.get('reference_checkpoint_sha256')
    matches = expected is not None and reference.get('checkpoint_sha256') == expected
    print(' Rhythm reference:', dict(path=str(reference_path), checkpoint_matches=matches,
                                    finite=reference.get('finite')))
    for name in ('RMDL', 'RMDR'):
        cell = rhythm.get('cells', {}).get(name)
        print(' Rhythm', name + ':', json.dumps(cell if cell is not None else 'unavailable', sort_keys=True))
        # These are the actual reference measurements used by the worker, not
        # reconstructed claims about why its relative gate failed.
        parent = reference.get('cells', {}).get(name) if matches else None
        print(' Parent', name + ':', json.dumps(parent if parent is not None else 'unavailable', sort_keys=True))
    print(' Details are saved evidence; missing fields are unavailable. No gates changed.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path)
    p.add_argument('--details',action='store_true',help='Show saved parameter errors, bounds, selection and per-cell rhythm evidence; no simulations')
    a=p.parse_args()
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
            if a.details:print_details(run, checks)
    if (run/'budget_receipt.json').exists():print('Budget:',json.loads((run/'budget_receipt.json').read_text()))
    print('No model promoted; rhythm and atlas checks do not establish body gait, sensory start/stop or swimming.')


if __name__=='__main__':main()
