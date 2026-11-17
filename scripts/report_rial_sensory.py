"""Report sensory fitting and automatic audit without importing JAX."""
import argparse,json
from pathlib import Path


def audit_details(run,summary):
    import numpy as np
    result=dict(scope='Read-only saved-data details; no changed acceptance criteria or new simulation',
                experiment=summary.get('experiment'),training_variant=summary.get('training_variant','nominal'),
                response_aggregation=summary.get('response_aggregation','mean'),operating_range={},recent_generations=[],near_bound_groups=[])
    for label in ('baseline','baseline_training','selected'):
        if not summary.get(label):continue
        row=summary.get(label) or {};paired=row.get('paired_features',{})
        if not paired or 'AIYL_release_q99' not in paired['no_cue']['active'][0]:
            result['operating_range'][label]=dict(applicable=False);continue
        windows=sum([paired['no_cue'][k] for k in ('active','response','late')],[])+paired['cue']['active']
        maximum=max(w['AIYL_release_q99'] for w in windows)
        result['operating_range'][label]=dict(variant=row.get('variant'),minimum_q01=min(w['AIYL_release_q01'] for w in windows),
            maximum_q99=maximum,upper_release_headroom=1-maximum,
            required_headroom=summary['final_target'],
            cue_changes=row['metrics'].get('relay_response_mean_changes'))
    for g in json.loads((run/'generation_history.json').read_text())[-10:]:
        evaluation=json.loads((Path(g['selected_evaluation'])/'summary.json').read_text());selected=evaluation['metrics']
        result['recent_generations'].append(dict(generation=g['generation'],target=g['target'],
            population_minimum=g['minimum_loss'],selected_loss=selected['loss'],
            selected_variant=evaluation.get('variant'),selected_metric_target=selected.get('target'),selected_metric_margin=selected.get('margin'),
            cue_changes=selected.get('relay_response_mean_changes'),range_pass=selected.get('relay_operating_range_passed')))
    audit=run.with_name(run.name+'-audit')
    parameters=json.loads((audit/'parameters.json').read_text())
    schema={g['name']:g for g in json.loads((run/'parameter_groups.json').read_text())}
    with np.load(run/'parameter_origin.npz',allow_pickle=False) as z:lower,upper=z['lower'],z['upper']
    with np.load(Path(summary['selected']['output'])/'coordinates.npz',allow_pickle=False) as z:offsets=z['offsets']
    for row in parameters['groups']:
        if not row['near_group_bound']:continue
        g=schema[row['name']];ix=g['indices']
        result['near_bound_groups'].append(dict(**row,local_log_bounds=[g['lower'],g['upper']],
            minimum_original_lower_room=float(np.min(offsets[ix]-lower[ix])),
            minimum_original_upper_room=float(np.min(upper[ix]-offsets[ix]))))
    meta=json.loads((run/'optimizer/final/metadata.json').read_text())
    result['optimizer']=dict(stage=meta['stage'],generation=meta['generation'],pending_points=len(meta.get('pending') or []))
    if (run/'partial_generation.json').exists():
        from collections import Counter
        partial=json.loads((run/'partial_generation.json').read_text())
        result['partial_generation_statuses']=dict(Counter(r['status'] for r in partial['evaluations']))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--release',action='store_true');p.add_argument('--relay',action='store_true');p.add_argument('--audit-details',action='store_true');a=p.parse_args();run=a.run
    if run is None:
        runs=[p for p in Path('/home/carsonc/r-elegans-results').glob('relay-range-*' if a.relay else 'rial-release-*' if a.release else 'rial-sensory-*') if p.is_dir() and not p.name.endswith('-audit') and not p.name.startswith('rial-sensory-audit-')]
        if not runs:raise SystemExit('No RIAL sensory run found')
        run=max(runs,key=lambda p:p.name)
    print('Run:',run)
    if not (run/'summary.json').exists():print('No final summary; inspect adjacent .log');return
    s=json.loads((run/'summary.json').read_text())
    if a.audit_details:
        print(json.dumps(audit_details(run,s),indent=2));return
    for k in ('experiment','execution_valid','validation_complete','completed_generations','stop_reason','failure','final_target','final_target_meaning',
              'objective','sensory_RIAL_release_component_qualified','sensory_relay_range_component_qualified','AIYL_to_RIAL_body_effect_mediation',
              'training_variant','response_aggregation',
              'partial_sensory_component_qualified','sensory_block_control','RIAL_chemical_replay_mediation',
              'RIAL_mediated_partial_component_qualified',
              'long_cue_sustained_attenuation','long_cue_RIAL_release_attenuation','long_cue_relay_response','full_start_stop_qualified'):print(k+':',s.get(k))
    print('Milestones:',[{k:r[k] for k in ('target','passed','checkpoint_sha256')} for r in s['milestones']])
    if (run/'optimization_budget.json').exists():print('Optimization budget:',json.loads((run/'optimization_budget.json').read_text()))
    for label,r in [('baseline',s['baseline']),('training_baseline',s.get('baseline_training')),('selected',s['selected'])]+[(v['variant']+'/'+v['grid']+'/'+v['mode'],v) for v in s['validation']]:
        if r:print(label,r['status'],r.get('metrics'),r.get('mediation'),r.get('checkpoint_sha256'))
    print('Numerical checks:',s['numerical_checks'])
    if (run/'budget_receipt.json').exists():print('Budget:',json.loads((run/'budget_receipt.json').read_text()))
    audit=run.with_name(run.name+'-audit')/'summary.json'
    if audit.exists():print('Post-run audit:',json.dumps(json.loads(audit.read_text()),indent=2))


if __name__=='__main__':main()
