"""Read saved worker/cohort validations; write spatial/cycle reports externally.

No simulation or training. Input trajectories/checkpoints are never modified.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from r_elegans.trajectory_analysis import analyze_trajectory,plot_analysis
from scripts.experiment_records import capture_run_record,file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='Cohort or individual worker directory')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--body-length',type=float,required=True,help='Length in position units: 0.001 for these SI liquid assays')
    parser.add_argument('--cases',nargs='+',help='Default: every recorded validation case')
    parser.add_argument('--plots',action='store_true',help='Optional matplotlib figures')
    args=parser.parse_args()
    if not math.isfinite(args.body_length) or args.body_length<=0:parser.error('Positive finite body length required')
    repo=Path(__file__).resolve().parents[1];run=args.run.expanduser().resolve();out=args.output_dir.expanduser().resolve()
    if out==repo or repo in out.parents or out==run or run in out.parents:
        parser.error('Keep analysis outputs outside source run and repository')
    source=json.loads((run/'summary.json').read_text())
    workers=[(f"seed-{w['seed']}",run/f"seed-{w['seed']}") for w in source['workers']] if 'workers' in source else [('worker',run)]
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,args)
    report=dict(source_run=str(run),body_length=args.body_length,body_length_source='explicit CLI; verify against recorded mechanics',
        summary_sha256=file_sha256(run/'summary.json'),analysis='late half, tail-to-head amplitudes and equal-reference-joint-phase translation',
        limitations=['One joint phase is not proof of whole-shape closure; residuals reported.',
                     'Spectral phase is modulo a period and does not establish propagation causality.',
                     'No model, objective or acceptance criterion changed.'],workers=[])
    failed=False
    for label,worker in workers:
        summary_path=worker/'summary.json'
        if not summary_path.exists():
            report['workers'].append(dict(worker=label,status='missing_worker_summary'));failed=True;continue
        metadata=json.loads(summary_path.read_text());results=[]
        diagnostic='chunks' in metadata
        checkpoint=worker/('evaluated_network.npz' if diagnostic else 'network_best.npz')
        if not checkpoint.exists():
            report['workers'].append(dict(worker=label,status='missing_checkpoint'));failed=True;continue
        with np.load(checkpoint,allow_pickle=False) as z:ids=z['neuron_ids'].tolist()
        cases=[dict(case='diagnostic',dt_s=metadata['dt_s'],chunks=metadata['chunks'])] if diagnostic else metadata.get('validation',[])
        for case in cases:
            name=case['case']
            if args.cases and name not in args.cases:continue
            if name not in ('long','perturb-plus','perturb-minus','brain-refined','body-refined','held-voltage','diagnostic'):
                raise ValueError('Unexpected validation case '+str(name))
            path=worker/f'validation-{name}.npz';destination=out/label/name
            destination.mkdir(parents=True,exist_ok=False)
            try:
                dt=float(case['dt_s'])
                files=[worker/f for f in case['chunks']] if diagnostic else [path]
                if not files:raise ValueError('No recorded trajectory files')
                if diagnostic and any(Path(f).name!=f for f in case['chunks']):raise ValueError('Invalid chunk filename')
                fields=('voltage_mV','body_joint_angles','body_muscle_activation','body_position','body_heading')
                arrays=[[] for _ in fields]
                for file in files:
                    with np.load(file,allow_pickle=False) as z:
                        for values,field in zip(arrays,fields):values.append(z[field])
                arrays=[np.concatenate(values) if len(values)>1 else values[0] for values in arrays]
                metrics,derived=analyze_trajectory(*arrays,dt=dt,body_length=args.body_length,neuron_ids=ids)
                del arrays
                record=dict(case=name,status='analyzed',trajectory_files={str(file):file_sha256(file) for file in files},metrics=metrics)
                np.savez_compressed(destination/'cycle_samples.npz',phase_times_s=derived['phase_times_s'],
                    phase_forward_body_lengths=derived['phase_forward_body_lengths'],
                    cycle_speed_body_lengths_s=derived['cycle_speed_body_lengths_s'])
                if args.plots:
                    try:
                        plot_analysis(metrics,derived,destination/'trajectory.png',f'{run.name} / {label} / {name}')
                        record['figure']=str(destination/'trajectory.png')
                    except ImportError as e:record['plot_unavailable']=str(e)
                (destination/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
                results.append(record)
                lags=[abs(x['lag_modulo_period_s'])*1000 for x in metrics['spectral_phase'] if x['coherence']>=.8]
                compact=dict(worker=label,case=name,anterior_variance_fraction=metrics['anterior_three_joint_variance_fraction'],
                    shape_pc1_fraction=metrics['shape_pca_variance_ratios'][0],active_joint_count=len(metrics['spectral_phase']),
                    maximum_active_lag_ms=max(lags) if lags else None,
                    frequency_hz=metrics['dominant_frequency_hz'],raw_min_speed=min(metrics['raw_forward_block_speeds_body_lengths_s']),
                    equal_phase_speed=metrics['equal_phase_regression_speed_body_lengths_s'],positive_cycle_fraction=metrics['positive_cycle_fraction'],
                    shape_closure_rms_rad=metrics['median_shape_closure_rms_rad'])
                print(json.dumps(compact,allow_nan=False),flush=True)
            except (OSError,ValueError,KeyError) as e:
                results.append(dict(case=name,status='analysis_failed',error=str(e)));failed=True
                print(json.dumps(dict(worker=label,case=name,error=str(e))),flush=True)
        if not results:failed=True
        report['workers'].append(dict(worker=label,checkpoint_sha256=file_sha256(checkpoint),
            source_summary_sha256=file_sha256(summary_path),cases=results))
    report['status']='incomplete' if failed else 'completed'
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Analysis: '+str(out),flush=True)
    return 2 if failed else 0


if __name__=='__main__':raise SystemExit(main())
