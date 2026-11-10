"""Assess saved qualified tail winner across the whole body;no simulation."""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
from r_elegans.full_body_assessment import regional_amplitudes,write_motion_preview,full_body_checks
from r_elegans.trajectory_analysis import analyze_trajectory,plot_analysis
from scripts.autonomous_warm_start import read_warm_start
from scripts.experiment_records import capture_run_record,file_sha256


CASES=('long','perturb-plus','perturb-minus','brain-refined','body-refined','held-voltage')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--body-length',type=float,default=.001,help='SI liquid assay length in meters;verify against recorded model')
    parser.add_argument('--preview-seconds',type=float,default=5.)
    parser.add_argument('--allow-unqualified',action='store_true',help='Explicit local pipeline test override,not qualification')
    args=parser.parse_args()
    if not all(math.isfinite(x) and x>0 for x in (args.body_length,args.preview_seconds)):parser.error('Positive finite length/preview duration required')
    repo=Path(__file__).resolve().parents[1];source=args.worker.expanduser().resolve();out=args.output_dir.expanduser().resolve()
    if out==repo or repo in out.parents or out==source or source in out.parents:parser.error('Output must be outside repository and source worker')
    warm=read_warm_start(source);metadata=warm['summary']
    if metadata.get('posterior_motor_focus') is not True:parser.error('Expected posterior-motor worker')
    if metadata.get('passed') is not True and not args.allow_unqualified:parser.error('Qualified worker required;override is for explicit pipeline tests only')
    cases={row['case']:row for row in metadata.get('validation',[])}
    if set(cases)!=set(CASES) or len(metadata['validation'])!=len(CASES):parser.error('Expected all six unique validation cases')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,args)
    reference=out/'parameter_reference';reference.mkdir()
    copies={'network_best.npz':'network.npz','source_inputs/parent_actor.npz':'parent_actor.npz',
            'source_inputs/network.npz':'original_anchor.npz','source_inputs/inputs.json':'original_input_manifest.json',
            'summary.json':'worker_summary.json','fit_policy.json':'fit_policy.json',
            'search_state.json':'search_state.json','allowed_parameters.npz':'allowed_parameters.npz','best.json':'best.json'}
    copied={}
    for original,destination in copies.items():
        before=file_sha256(source/original);shutil.copy2(source/original,reference/destination)
        if file_sha256(reference/destination)!=before:raise ValueError('Reference copy checksum mismatch')
        copied[original]=dict(copy=destination,sha256=before)
    (reference/'reference_manifest.json').write_text(json.dumps(dict(source_worker=str(source),files=copied,
        scope='Selected parameters and essential metadata only;not a complete raw trajectory/recursive-lineage backup'),indent=2)+'\n')
    with np.load(source/'network_best.npz',allow_pickle=False) as z:
        ids=z['neuron_ids'].tolist();zero_current=bool(np.all(z['fixed_bias_pA']==0))
    if not zero_current:raise ValueError('Zero-current winner expected')
    results=[]
    for name in CASES:
        entry=cases[name];dt=float(entry['dt_s'])
        if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid recorded dt')
        path=source/f'validation-{name}.npz';before=file_sha256(path)
        with np.load(path,allow_pickle=False) as z:
            voltage=z['voltage_mV'];q=z['body_joint_angles'];drive=z['body_muscle_activation'];position=z['body_position'];heading=z['body_heading']
        metrics,derived=analyze_trajectory(voltage,q,drive,position,heading,dt=dt,body_length=args.body_length,neuron_ids=ids)
        regions=regional_amplitudes(q,drive)
        recorded=entry['metrics']
        np.testing.assert_allclose(regions['regional_joint_median_p2p_rad'],recorded['regional_joint_median_p2p_rad'],rtol=1e-6,atol=1e-10,
            err_msg='Saved trajectory amplitudes differ from source summary')
        destination=out/name;destination.mkdir()
        plot_analysis(metrics,derived,destination/'trajectory.png',f'{source.name} / {name}')
        np.savez_compressed(destination/'cycle_samples.npz',phase_times_s=derived['phase_times_s'],
            phase_forward_body_lengths=derived['phase_forward_body_lengths'],cycle_speed_body_lengths_s=derived['cycle_speed_body_lengths_s'])
        row=dict(case=name,trajectory_sha256=before,dt_s=dt,recording_seconds=len(q)*dt,regions=regions,
            recorded_tail_qualified=recorded.get('posterior_motor_qualified'),
            recorded_full_body_amplitude_qualified=recorded.get('body_amplitude_qualified'),
            recorded_valid=recorded.get('valid'),recorded_neural_rhythms=recorded.get('sustained_neural_rhythms'),trajectory_analysis=metrics,
            figure=str(destination/'trajectory.png'))
        if name=='long':
            row['preview']=write_motion_preview(out/'motion.html',q,position,heading,dt=dt,body_length=args.body_length,
                seconds=args.preview_seconds,title=f'{source.parent.name} / {source.name} / saved long validation')
        if file_sha256(path)!=before:raise ValueError('Source trajectory changed during analysis')
        (destination/'report.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');results.append(row)
        print(json.dumps(dict(case=name,bends_tail_middle_head=regions['regional_joint_median_p2p_rad'],
            filtered_drive_tail_middle_head=regions['regional_filtered_drive_median_p2p'],
            all_three_amplitude_targets_met=regions['all_three_regional_amplitude_targets_met'],
            recorded_full_body_amplitude_qualified=row['recorded_full_body_amplitude_qualified'],
            strongest_joint=metrics['strongest_joint'],shape_pc1_fraction=metrics['shape_pca_variance_ratios'][0],
            equal_phase_speed_BL_s=metrics['equal_phase_regression_speed_body_lengths_s']),allow_nan=False),flush=True)
        del voltage,q,drive,position,heading,derived
    after=read_warm_start(source)
    if after['essential_sha256']!=warm['essential_sha256']:raise ValueError('Parent changed during analysis')
    active=[r for r in results if r['case']!='held-voltage']
    report=dict(experiment='E044',status='completed',source_worker=str(source),source_passed=metadata['passed'],
        explicit_unqualified_test_override=args.allow_unqualified,source_essential_sha256=warm['essential_sha256'],
        checkpoint_sha256=warm['essential_sha256']['network_best.npz'],injected_currents_zero=zero_current,
        body_length=args.body_length,body_length_source='CLI value;SI liquid twelve-rod assay default0.001m',
        cases=results,full_body_checks=full_body_checks(results,metadata['best_training']['regional_joint_median_p2p_rad']),
        all_active_recorded_full_body_amplitude_qualified=all(r['recorded_full_body_amplitude_qualified'] is True for r in active),
        minimum_active_regional_joint_medians=np.min([r['regions']['regional_joint_median_p2p_rad'] for r in active],axis=0).tolist(),
        limitations=['No parameters,simulation or acceptance criteria changed.',
                     'Tail-only qualification does not establish head/middle retention;reported separately.',
                     'Single-reference-joint equal phase does not prove whole-shape cycle closure.',
                     'Translation and spectral phase are descriptive,not locomotion qualification.',
                     'Preview is subsampled;metrics use all samples. Visual gain never alters world geometry.',
                     'Parameter-reference copy on same server is not an independent raw-lineage backup.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    files={str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'artifact_manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    print(json.dumps(dict(checkpoint_sha256=report['checkpoint_sha256'],
        full_body_amplitude_assay_pass=report['full_body_checks']['full_body_amplitude_assay_pass'],
        all_active_full_body_amplitude_qualified=report['all_active_recorded_full_body_amplitude_qualified'],
        minimum_active_bends_tail_middle_head=report['minimum_active_regional_joint_medians'],
        report=str(out/'report.json'),motion_preview=str(out/'motion.html'),figure=str(out/'long/trajectory.png'))),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
