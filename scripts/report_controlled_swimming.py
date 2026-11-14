"""Read the latest sensory-control/swimming pipeline; no simulation or writes."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args()
    if a.run:run=a.run.expanduser().resolve()
    else:
        runs=[r for r in Path('/home/carsonc/r-elegans-results').glob('controlled-swimming-*') if r.is_dir()]
        if not runs:p.error('No controlled-swimming run found')
        run=max(runs,key=lambda r:r.name)
    print('Run:',run)
    if (run/'budget_receipt.json').exists():
        b=json.loads((run/'budget_receipt.json').read_text())
        print('Elapsed minutes:',round(b['total_elapsed_seconds']/60,2))
        print({k:b.get(k) for k in ('all_steps_finished','validation_complete','timed_out','exit_code')})
    if (run/'configuration.json').exists():
        config=json.loads((run/'configuration.json').read_text())
        reserve=config.get('validation_reserve_seconds_each')
        if reserve is not None:print('Validation reserve minutes per stage:',round(reserve/60,2))
    if not (run/'summary.json').exists():print('No final summary yet; inspect the log.');return
    s=json.loads((run/'summary.json').read_text())
    print({k:s.get(k) for k in ('control_qualified','swimming_qualified','swimming_blocked','stop_reason')})
    for phase in s.get('phases',[]):
        stage=phase['stage'];print('\nStage:',stage,'selected:',phase['selected'],'passed:',phase['passed'])
        folder='control' if phase.get('reused_same_checkpoint_motion_validation') else stage
        gate=json.loads((run/folder/'control_gate.json').read_text())
        print('Checkpoint:',gate['checkpoint_sha256'])
        print({k:gate.get(k) for k in ('passed','complete','positive_cases_passed','sensory_specificity_passed','negative_controls_passed','causal_control_groups','refinement_passed')})
        for row in gate['cases']:
            m=row['metrics'];speeds=[w['propulsion']['speed_BL_s'] for w in m['on']]
            print(row['case'],dict(control=m['control_passed'],swimming=m['swimming_passed'],
                active_hz=[w['frequency_hz'] for w in m['on']],
                maximum_rest_range_rad=max(w['maximum_raw_joint_range_rad'] for w in m['off']),
                forward_speeds_BL_s=speeds))
        nominal=next((r for r in gate['cases'] if r['case']=='nominal'),None)
        if nominal:print('Transition latencies:',nominal['metrics']['transition_latencies'])
        print('Repeated-cue motion preview:',run/folder/'validate/repeat/motion.html')
    if not s.get('phases'):
        for stage,runs in s.get('searches',{}).items():
            for r in runs:
                result=r.get('result') or {}
                print(stage,r['path'],'exit:',r['exit_code'],'epoch:',result.get('epoch'),
                      'nominal passed:',result.get('nominal_passed'),'iterations:',result.get('iterations'))
                worker=Path(r['path'])
                baseline=worker/'baseline.json'
                rows=[('baseline (source no-cue for control)',json.loads(baseline.read_text()))] if baseline.exists() else []
                rows.append(('best cued',result.get('best',{}).get('metrics',{})))
                for label,m in rows:
                    print(label,dict(stop_loss=m.get('stop_loss'),safety=m.get('safety_passed'),
                        initial_movement=m.get('initial_movement_passed'),
                        rest_range_rad=[w.get('maximum_raw_joint_range_rad') for w in m.get('off',[])],
                        active_amplitudes=[w.get('amplitudes') for w in m.get('on',[])],
                        active_hz=[w.get('frequency_hz') for w in m.get('on',[])]))
                print('Checkpoint:',result.get('checkpoint_sha256'))
                history=worker/'history.jsonl'
                if history.exists():
                    records=[json.loads(line) for line in history.read_text().splitlines() if line.strip()]
                    print('Accepted:',sum(bool(row.get('accepted')) for row in records),
                          'proposal errors:',sum(bool(row.get('error')) for row in records))


if __name__=='__main__':main()
