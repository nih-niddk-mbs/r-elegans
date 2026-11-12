"""E067 fresh bounded connection-only search for small unpaced cyclic bends."""
import argparse
import json
import math
import os
import re
from pathlib import Path
import shutil
import time

from scripts.experiment_records import capture_run_record,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--minutes',type=float,default=240.)
    p.add_argument('--seed',type=int,default=6700)
    p.add_argument('--iterations',type=int,default=1000000)
    p.add_argument('--sigma',type=float,default=.025)
    p.add_argument('--train-seconds',type=float,default=60.)
    p.add_argument('--window-seconds',type=float,default=15.)
    p.add_argument('--long-seconds',type=float,default=300.)
    p.add_argument('--software-smoke',action='store_true')
    a=p.parse_args()
    if (not all(math.isfinite(x) and x>0 for x in (a.minutes,a.sigma,a.train_seconds,a.window_seconds,a.long_seconds))
        or a.iterations<1 or a.seed<0 or a.train_seconds<12 or a.long_seconds<a.train_seconds
        or a.train_seconds<2*a.window_seconds+4):p.error('Positive finite budget and valid two-window duration required')
    if not a.software_smoke and (a.train_seconds!=60 or a.window_seconds!=15 or a.long_seconds<300):
        p.error('Scientific protocol is60s/two15s windows with>=300s final validation')
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64',False)
    from scripts.connection_amplitude_setup import setup
    from scripts.compare_rhythm_worker import verify_inputs
    from scripts.assess_wet_surface import write_json
    from r_elegans.brain.conductance_network import save_network
    from r_elegans.connection_amplitude import TARGETS,amplitude_score,propose_connections
    root=a.inputs.resolve();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    for source in (root,repo):
        if out==source or source in out.parents or out in source.parents:p.error('Separate external output required')
    d=setup(root);base=d['base'];spec=d['spec'];origin=d['origin']
    if d['manifest'].get('protocol')!='connection-amplitude':raise ValueError('Verified connection-amplitude inputs required')
    if d['manifest'].get('software_smoke') and not a.software_smoke:
        raise ValueError('Short local source requires explicit software-smoke mode')
    out.mkdir(parents=True,exist_ok=False)
    os.environ['JAX_ENABLE_X64']='true';capture_run_record(out,repo,a)
    shutil.copy2(root/'candidate.npz',out/'parameter_reference.npz')
    np.savez_compressed(out/'coordinate_reference.npz',origin=origin,lower=d['lower'],upper=d['upper'],
                        chemical_mask=spec.chemical,gap_mask=spec.gap)
    rng=np.random.default_rng(a.seed);x=origin.copy();iteration=0;selected=0;accepted_count=0
    write_json(out/'initial_rng_state.json',rng.bit_generator.state)
    settings=dict(experiment='E067',arguments=vars(a),status='planned_before_execution',
        parent_sha256=d['manifest']['candidate_sha256'],input_manifest_sha256=file_sha256(root/'manifest.json'),
        reference_sha256=file_sha256(out/'parameter_reference.npz'),origin_sha256=file_sha256(out/'coordinate_reference.npz'),
        initial_state_sha256=d['fingerprint'],targets_rad=TARGETS.tolist(),trainable=spec.counts,
        fixed=['all channel scales and kinetics','leak/capacitance/reversals','topology/release','NMJ/muscles/mechanics','sensory maps/gains'],
        maximum_factor_from_original_anchor=2.,inherited_roundoff_slack_log=d['slack'],
        reference_motor_neuron=d['manifest']['reference_motor_neuron'],
        objective='Mean plus worst log deficit of true regional cyclic excursions over two windows, plus .001 mean squared change from hybrid',
        initialization='Fresh seeded greedy search; standard common source state for every rollout, no state reconstruction',
        time_budget='Baseline/search240min by default; full validation follows budget',
        external_drive='No ligand cue or injected bias/pacing; actual-body proprioception active',
        no_automatic_promotion=True)
    settings=json.loads(json.dumps(settings,default=str))
    write_json(out/'configuration.json',settings)
    ids=d['model'].neuron_ids;ref=ids.index(settings['reference_motor_neuron'])
    focus=[i for i,n in enumerate(ids) if re.fullmatch(r'(DA|DB|DD|VA|VB|VD|AS)\d+',n)
           or n.startswith(('RMD','SMD','SMB','AVA','AVB','AVD','AVE','PVC'))]
    run=d['factory']();times=np.linspace(0,a.train_seconds,121)
    started=time.monotonic()
    baseline_trace=run(d['decode'](x),times,d['gains'])
    baseline=amplitude_score(baseline_trace,ref,window_seconds=a.window_seconds)
    if not baseline['neural_guard'] or not baseline['safety_passed']:
        raise ValueError('Parent does not meet neural/safety guard in training windows')
    # Verify every available source reference timestamp before any proposal.
    with np.load(root/'reference-training-sham.npz',allow_pickle=False) as z:
        use=baseline_trace['dense_time_s']<=z['dense_time_s'][-1]+1e-12
        if not np.array_equal(baseline_trace['dense_time_s'][use],z['dense_time_s']):
            raise ValueError('Training/reference grids differ')
        fields=('dense_voltage_mV','dense_calcium_uM','dense_muscles','dense_command','dense_filtered_drive','dense_joints')
        errors={k:float(np.max(np.abs(baseline_trace[k][use]-z[k]))) for k in fields}
        if any(errors[k]>5e-10*(1+float(np.max(np.abs(z[k])))) for k in fields):
            raise ValueError('Training baseline changed source trajectory')
    write_json(out/'baseline_reference_check.json',dict(errors=errors,passed=True))
    profiles=baseline['reference_motor_profiles']
    def evaluate(coords):
        trace=run(d['decode'](coords),times,d['gains'])
        metrics=amplitude_score(trace,ref,window_seconds=a.window_seconds,baseline_profiles=profiles)
        prior=.001*float(np.mean((coords-origin)**2))
        return dict(loss=metrics['data_loss']+prior,prior=prior,metrics=metrics),trace
    best=dict(loss=baseline['data_loss'],prior=0.,metrics=baseline)
    def save_selected(number,trace):
        path=out/'selected'/f'iteration-{number:06d}';path.mkdir(parents=True)
        save_network(path/'network.npz',d['decode'](x),d['fixed'],d['config'])
        np.savez_compressed(path/'optimizer_state.npz',offsets=x,iteration=number)
        write_json(path/'rng_state.json',rng.bit_generator.state)
        write_json(path/'metrics.json',best)
        np.savez_compressed(path/'trajectory.npz',**trace)
        shutil.copy2(path/'network.npz',out/'network_best.npz')
        shutil.copy2(path/'optimizer_state.npz',out/'best_state.npz')
    save_selected(0,baseline_trace)
    del baseline_trace
    write_json(out/'baseline.json',best)
    write_json(out/'progress.json',dict(iteration=0,best_loss=best['loss'],minimum_cycle_excursions_rad=baseline['minimum_cycle_excursions_rad']))
    print(json.dumps(dict(seed=a.seed,baseline=best)),flush=True)
    with (out/'history.jsonl').open('w') as log:
        for number in range(1,a.iterations+1):
            if time.monotonic()-started>=a.minutes*60:break
            candidate,proposal=propose_connections(x,spec,rng,a.sigma,d['lower'],d['upper'],focus)
            error=None;result=None;accepted=False
            try:
                result,trace=evaluate(candidate)
                accepted=(result['metrics']['neural_guard'] and result['metrics']['safety_passed'] and result['loss']<best['loss'])
            except (ValueError,RuntimeError,FloatingPointError) as e:error=str(e)
            if accepted:
                x=candidate;best=result;selected=number;accepted_count+=1;save_selected(number,trace)
            iteration=number
            row=dict(iteration=number,accepted=bool(accepted),proposal=proposal,result=result,error=error,
                     best_loss=best['loss'],elapsed_seconds=time.monotonic()-started,rng_after=rng.bit_generator.state)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            np.savez_compressed(out/'optimizer_state.npz',offsets=x,iteration=iteration)
            write_json(out/'rng_state.json',rng.bit_generator.state)
            write_json(out/'progress.json',dict(iteration=iteration,accepted_proposals=accepted_count,
                best_loss=best['loss'],minimum_cycle_excursions_rad=best['metrics']['minimum_cycle_excursions_rad']))
            print(json.dumps(row,allow_nan=False),flush=True)
            if result is not None:del trace
    if iteration==0:
        np.savez_compressed(out/'optimizer_state.npz',offsets=x,iteration=0)
        write_json(out/'rng_state.json',rng.bit_generator.state)
    search_elapsed=time.monotonic()-started
    params=d['decode'](x)
    save_network(out/'network_final.npz',params,d['fixed'],d['config'])
    write_json(out/'best.json',dict(selected_iteration=selected,result=best))
    validation=[]
    cases=[('ordinary',a.train_seconds,.002,.1,0.,False),
           ('voltage-plus',a.train_seconds,.002,.1,1.,False),('voltage-minus',a.train_seconds,.002,.1,-1.,False),
           ('neural-half',a.train_seconds,.002,.05,0.,False),('body-half',a.train_seconds,.001,.1,0.,False),
           ('long',a.long_seconds,.002,.1,0.,False),('held-voltage',a.train_seconds,.002,.1,0.,True)]
    for name,duration,dt,brain_step,perturb,held in cases:
        path=out/'validation'/name;path.mkdir(parents=True)
        trace=d['factory'](dt,brain_step,perturb,held)(params,np.linspace(0,duration,121),d['gains'])
        window=duration/4 if name=='long' else a.window_seconds
        metrics=amplitude_score(trace,ref,window_seconds=window,baseline_profiles=profiles)
        np.savez_compressed(path/'trajectory.npz',**trace)
        write_json(path/'metrics.json',metrics)
        validation.append(dict(case=name,simulation_seconds=duration,metrics=metrics))
        print(json.dumps(dict(validation=validation[-1])),flush=True)
    rows={r['case']:r['metrics'] for r in validation};base_metrics=rows['ordinary'];refinement=[]
    for name in ('neural-half','body-half'):
        a0=np.asarray(base_metrics['minimum_cycle_excursions_rad']);a1=np.asarray(rows[name]['minimum_cycle_excursions_rad'])
        body_ok=bool(np.all(abs(a1-a0)<=.2*np.maximum(a0,1e-7)))
        motor_ok=all(abs(p['frequency_hz']-q['frequency_hz'])<=.1*p['frequency_hz']
                     for p,q in zip(base_metrics['reference_motor_profiles'],rows[name]['reference_motor_profiles'])
                     if p['frequency_hz'] is not None and q['frequency_hz'] is not None)
        motor_ok=bool(motor_ok and rows[name]['neural_guard'])
        refinement.append(dict(case=name,body_cycle_amplitudes_consistent=body_ok,motor_frequency_consistent=motor_ok,passed=body_ok and motor_ok))
    held=rows['held-voltage']
    negative=bool(all(p['peaks']==0 and p['troughs']==0 and p['amplitude_mV']<=1e-9 for p in held['reference_motor_profiles'])
                  and np.max(held['window_cycle_excursions_rad'])<=1e-7)
    active=[r for n,r in rows.items() if n!='held-voltage']
    passed=bool(all(r['target_passed'] for r in active) and negative and all(r['passed'] for r in refinement))
    verify_inputs(root)
    frozen=bool(np.array_equal(np.asarray(params.log_channel_scale),np.asarray(base.log_channel_scale)))
    if not frozen:raise ValueError('Channel scales changed')
    from r_elegans.brain.conductance_network import load_network
    saved,saved_fixed,saved_config=load_network(out/'network_final.npz')
    if (saved_config!=d['config'] or any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(
            jax.tree_util.tree_leaves((params,d['fixed'])),jax.tree_util.tree_leaves((saved,saved_fixed))))):
        raise ValueError('Final checkpoint does not match parameters/fixed fields')
    summary=dict(experiment='E067',status='completed',seed=a.seed,passed=passed,requested_search_minutes=a.minutes,
        search_elapsed_seconds=search_elapsed,total_elapsed_seconds=time.monotonic()-started,
        stop_reason='search_time_budget' if iteration<a.iterations else 'iterations',iterations=iteration,
        selected_iteration=selected,accepted_proposals=accepted_count,baseline=baseline,best=best,
        validation=validation,refinement=refinement,held_voltage_control_passed=negative,
        channel_scales_unchanged=frozen,all_fixed_fields_unchanged=True,
        connection_coordinates_changed=int(np.count_nonzero(x-origin)),parent_sha256=settings['parent_sha256'],
        checkpoint_sha256=file_sha256(out/'network_best.npz'),source_inputs_unchanged=True,
        software_smoke=a.software_smoke,biological_fidelity_qualified=False,locomotion_qualified=False,
        scope='Small raw cyclic-amplitude rung with frozen channels; no phase/propulsion/biological qualification')
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,allow_nan=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
