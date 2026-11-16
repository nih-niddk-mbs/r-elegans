"""Report E085 without loading trajectories or importing JAX."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args();run=a.run
    if run is None:
        runs=[p for p in Path('/home/carsonc/r-elegans-results').glob('rial-sensory-*') if p.is_dir()]
        if not runs:raise SystemExit('No RIAL sensory run found')
        run=max(runs,key=lambda p:p.name)
    print('Run:',run)
    if not (run/'summary.json').exists():print('No final summary; inspect adjacent .log');return
    s=json.loads((run/'summary.json').read_text())
    for k in ('execution_valid','validation_complete','completed_generations','stop_reason','failure','final_target',
              'partial_sensory_component_qualified','sensory_block_control','RIAL_chemical_replay_mediation',
              'RIAL_mediated_partial_component_qualified',
              'long_cue_sustained_attenuation','full_start_stop_qualified'):print(k+':',s.get(k))
    print('Milestones:',[{k:r[k] for k in ('target','passed','checkpoint_sha256')} for r in s['milestones']])
    for label,r in [('baseline',s['baseline']),('selected',s['selected'])]+[(v['variant']+'/'+v['grid']+'/'+v['mode'],v) for v in s['validation']]:
        if r:print(label,r['status'],r.get('metrics'),r.get('mediation'),r.get('checkpoint_sha256'))
    print('Numerical checks:',s['numerical_checks'])
    if (run/'budget_receipt.json').exists():print('Budget:',json.loads((run/'budget_receipt.json').read_text()))


if __name__=='__main__':main()
