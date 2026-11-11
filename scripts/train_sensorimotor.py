"""E055 stage1: finite ligand cue -> sensory neurons -> reversible body bend.

Fresh bounded search from a verified paired fit, with the original anchor.
Later curriculum stages are specified separately and are not run by this script.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import time

from scripts.experiment_records import capture_run_record,file_sha256
from scripts.fit_paired_recordings import write_json


def verified_parent(source):
    """Require saved offsets, original anchor, anatomy and fixed fields to agree."""
    import jax
    import numpy as np
    from scripts.assess_wet_surface import verify_reference
    from scripts.paired_resume import read_resume
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets
    from r_elegans.rl.full_network_actor import actor_from_arrays
    source=Path(source).resolve()
    cfg=json.loads((source/'configuration.json').read_text())
    _,anchor,readout=verify_reference(source/'inputs/parent_lineage')
    initial,original_fixed,network_config=load_network(anchor)
    fixed,spec=joint_fit_reference(original_fixed)
    offsets,iteration,_,summary=read_resume(source,cfg,spec.size)
    actual,actual_fixed,actual_config=load_network(source/'network_best.npz')
    expected=apply_joint_offsets(initial,offsets,spec)
    if (actual_config!=network_config or asdict(network_config)!=cfg['network_config']
            or file_sha256(anchor)!=cfg['network_sha256'] or file_sha256(readout)!=cfg['readout_sha256']
            or any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(
                jax.tree_util.tree_leaves((expected,fixed)),jax.tree_util.tree_leaves((actual,actual_fixed))))):
        raise ValueError('Parent checkpoint, offsets, fixed fields or original reference disagree')
    manifest=json.loads((source/'inputs/manifest.json').read_text())
    asset=Path(__file__).resolve().parents[1]/'r_elegans/assets/runtime_model_v1.json'
    if file_sha256(asset)!=manifest['canonical_model_asset_sha256']:raise ValueError('Runtime anatomy changed')
    if np.any(np.asarray(original_fixed.bias_pA)!=0):raise ValueError('Zero-bias parent required')
    with np.load(readout,allow_pickle=False) as z:actor=actor_from_arrays(z)
    return initial,fixed,network_config,spec,offsets,actor,cfg,summary,iteration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source_group=p.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--parent-run',type=Path)
    source_group.add_argument('--resume-run',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--minutes',type=float,default=20.)
    p.add_argument('--iterations',type=int,default=100000)
    p.add_argument('--seed',type=int,default=5500)
    p.add_argument('--sigma',type=float,default=.03)
    p.add_argument('--bend-target-rad',type=float,default=.00003)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--brain-step-ms',type=float,default=.1)
    args=p.parse_args()
    if args.iterations<1 or not all(math.isfinite(v) and v>0 for v in
            (args.minutes,args.sigma,args.dt,args.brain_step_ms,args.bend_target_rad)) or args.dt>.002 or args.brain_step_ms>.1:
        p.error('Positive settings; dt<=.002s and neural substep<=.1ms required')
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION','highest')
    import numpy as np
    from r_elegans.model import load_builtin_model
    from r_elegans.brain.paired_fitting import make_rollout
    from r_elegans.brain.joint_autonomous_fit import apply_joint_offsets,joint_proposal,frozen_joint_fields_unchanged
    from r_elegans.brain.conductance_network import save_network,load_network
    from r_elegans.body.embodied_sensory import EmbodiedSensoryConfig
    from r_elegans.sensorimotor_curriculum import ResponseProtocol,cue_world,response_metrics
    from r_elegans.data.paths import get_data_root
    from r_elegans.full_body_assessment import write_motion_preview
    from r_elegans.paired_diagnostics import signal_chain
    resume=args.resume_run.resolve() if args.resume_run else None
    source=resume/'parent_run' if resume else args.parent_run.resolve()
    initial,fixed,config,spec,start_offsets,actor,cfg,parent_summary,parent_iteration=verified_parent(source)
    factor=cfg['arguments']['maximum_factor']
    if factor!=2.:raise ValueError('E055 requires unchanged original factor2 neural bounds')
    out=get_data_root(args.output_dir,must_exist=False)
    if out==source or source in out.parents or out in source.parents:
        p.error('Output must be separate from parent lineage')
    if resume and (out==resume or resume in out.parents or out in resume.parents):
        p.error('Continuation output must be separate from resume parent')
    if any(path.is_symlink() for path in source.rglob('*')):raise ValueError('No symlinks in parent archive')
    out.mkdir(parents=True,exist_ok=False)
    capture_run_record(out,Path(__file__).resolve().parents[1],args)
    shutil.copytree(source,out/'parent_run')
    if file_sha256(out/'parent_run/network_best.npz')!=parent_summary['checkpoint_sha256']:
        raise ValueError('Copied parent checksum mismatch')
    model=load_builtin_model();protocol=ResponseProtocol(bend_target_rad=args.bend_target_rad);sc=EmbodiedSensoryConfig()
    settings=dict(experiment='E055',stage='contract_relax',status='planned_before_execution',
        parent_run=str(source),parent_checkpoint_sha256=parent_summary['checkpoint_sha256'],
        parent_global_iteration=parent_iteration,original_anchor_sha256=cfg['network_sha256'],
        readout_sha256=cfg['readout_sha256'],maximum_factor=factor,protocol=asdict(protocol),
        base_sensory_config=asdict(sc),sensory_gain_order=['chemical','B_proprioception','DVA_proprioception'],
        world={key:np.asarray(value).tolist() for key,value in cue_world(protocol)._asdict().items()},
        sensory_gain_bounds=[.5,2.],trainable=spec.counts|dict(sensory_gains=3),
        fixed=['topology','leak','capacitance','channel/reporter/receptor kinetics','reversals',
            'sensory signs and receptive maps','NMJ','muscle','mechanics'],
        optimizer='Fresh greedy search and PCG64 RNG for NEW objective; not an E053 optimizer continuation',
        medium='wet-agar1.7% proxy; not calibrated mounted2% agar or a diacetyl reflex',
        stage_transition='No automatic advancement; review validation before next stage',
        arguments=vars(args))
    settings=json.loads(json.dumps(settings,default=str))
    rng=np.random.default_rng(args.seed)
    offsets=start_offsets.copy();sensory_offsets=np.zeros(3)
    initial_iteration=0;resume_summary=None
    if resume:
        from scripts.sensorimotor_resume import read_completed
        import jax
        offsets,sensory_offsets,initial_iteration,rng,resume_summary=read_completed(resume,settings,spec.size)
        expected=apply_joint_offsets(initial,offsets,spec)
        actual,actual_fixed,actual_config=load_network(resume/'network_best.npz')
        if actual_config!=config or any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(
                jax.tree_util.tree_leaves((expected,fixed)),jax.tree_util.tree_leaves((actual,actual_fixed)))):
            raise ValueError('Resume network, fixed fields and original-anchor offsets disagree')
        if any(path.is_symlink() for path in resume.rglob('*')):raise ValueError('Resume archive contains symlinks')
        shutil.copytree(resume,out/'resume_parent')
        if file_sha256(out/'resume_parent/network_best.npz')!=resume_summary['checkpoint_sha256']:
            raise ValueError('Copied resume checkpoint mismatch')
        settings['resume']=dict(parent=str(resume),checkpoint_sha256=resume_summary['checkpoint_sha256'],
            initial_iteration=initial_iteration,rng_sha256=file_sha256(resume/'rng_state.json'))
        settings['optimizer']='Completed greedy search continuation: same objective, original bounds and saved PCG64; not interrupted simulation resume'
    run_start_offsets=offsets.copy()
    write_json(out/'configuration.json',settings)
    write_json(out/'initial_rng_state.json',rng.bit_generator.state)
    times=np.linspace(0,protocol.duration_s,81)
    def factory(*,dose=1.,shift=0.,dt=None,held=False,sham=False):
        return make_rollout(fixed,config,actor,model,medium='wet-agar',dt=args.dt if dt is None else dt,
            brain_step_ms=args.brain_step_ms if dt is None else args.brain_step_ms/2,
            burn_in=0.,sampling='point',record_dense=True,hold_voltage=held,
            sensory_config=sc,chemical_world=cue_world(protocol,dose=dose,onset_shift=shift,sham=sham))
    active_run=factory();sham_run=factory(sham=True)
    def evaluate(x,s,run=active_run,reference=sham_run,shift=0.):
        params=apply_joint_offsets(initial,x,spec)
        if not frozen_joint_fields_unchanged(params,initial,spec):raise RuntimeError('Frozen fields changed')
        sham=reference(params,times,np.exp(s));active=run(params,times,np.exp(s))
        metrics=response_metrics(active,sham,model.neuron_ids,protocol,onset_shift=shift)
        prior=.001*float(np.mean(x*x))+.001*float(np.mean(s*s))
        return dict(total=metrics['total']+prior,prior=prior,metrics=metrics),dict(active=active,sham=sham)
    started=time.monotonic()
    baseline,traces=evaluate(offsets,sensory_offsets)
    if resume_summary and not math.isclose(baseline['total'],resume_summary['best']['total'],rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError('Re-evaluated resume baseline disagrees with saved best objective')
    best=baseline
    def save_selected(iteration,traces):
        destination=out/'selected'/f'iteration-{iteration:06d}';destination.mkdir(parents=True)
        save_network(destination/'network.npz',apply_joint_offsets(initial,offsets,spec),fixed,config)
        np.savez_compressed(destination/'search_state.npz',offsets=offsets,sensory_offsets=sensory_offsets,iteration=iteration)
        write_json(destination/'rng_state.json',rng.bit_generator.state)
        write_json(destination/'metrics.json',best)
        for label,trace in traces.items():np.savez_compressed(destination/(label+'-trajectory.npz'),**trace)
        shutil.copy2(destination/'network.npz',out/'network_best.npz')
        shutil.copy2(destination/'search_state.npz',out/'best_state.npz')
        write_json(out/'best.json',dict(iteration=iteration,result=best,
            sensory_gain_multipliers=np.exp(sensory_offsets).tolist()))
    save_selected(initial_iteration,traces)
    print(json.dumps(dict(baseline=baseline,parent_checkpoint_sha256=parent_summary['checkpoint_sha256'])),flush=True)
    completed=0;selected_iteration=initial_iteration
    with (out/'history.jsonl').open('w') as history:
        for run_iteration in range(1,args.iterations+1):
            if time.monotonic()-started>=args.minutes*60:break
            iteration=initial_iteration+run_iteration
            candidate=joint_proposal(offsets,spec,rng,args.sigma,factor)
            candidate_s=np.clip(sensory_offsets+rng.normal(0,args.sigma,3),-np.log(2),np.log(2))
            error=None;accepted=False;result=None
            try:
                result,traces=evaluate(candidate,candidate_s)
                accepted=result['total']<best['total']
            except (ValueError,RuntimeError,FloatingPointError) as exc:error=str(exc)
            if accepted:
                offsets=candidate;sensory_offsets=candidate_s;best=result;selected_iteration=iteration
                save_selected(iteration,traces)
            completed=run_iteration
            row=dict(iteration=iteration,run_iteration=run_iteration,accepted=accepted,result=result,error=error,
                elapsed_seconds=time.monotonic()-started)
            history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
            write_json(out/'rng_state.json',rng.bit_generator.state)
            np.savez_compressed(out/'search_state.npz',offsets=offsets,sensory_offsets=sensory_offsets,iteration=iteration)
            print(json.dumps(row,allow_nan=False),flush=True)
    if completed==0:
        write_json(out/'rng_state.json',rng.bit_generator.state)
        np.savez_compressed(out/'search_state.npz',offsets=offsets,sensory_offsets=sensory_offsets,iteration=initial_iteration)
    save_network(out/'network_final.npz',apply_joint_offsets(initial,offsets,spec),fixed,config)
    # This greedy optimizer's final state IS its best state; retain both explicitly.
    write_json(out/'sensory_parameters.json',dict(base=asdict(sc),gain_multipliers=np.exp(sensory_offsets).tolist(),
        chemical_gain_nS=float(sc.chemical_gain_nS*np.exp(sensory_offsets[0])),
        b_bend_gain_nS=float(sc.b_bend_gain_nS*np.exp(sensory_offsets[1])),
        dva_gain_nS=float(sc.dva_gain_nS*np.exp(sensory_offsets[2]))))
    validation=[];coarse_difference=None;refinement=None
    for name,run,reference,shift in [('active',active_run,sham_run,0.),
            ('dose-low',factory(dose=.8),sham_run,0.),('dose-high',factory(dose=1.2),sham_run,0.),
            ('onset-shift',factory(shift=.5),sham_run,.5),
            ('refined',factory(dt=args.dt/2),factory(dt=args.dt/2,sham=True),0.),
            ('chemical-blocked',active_run,sham_run,0.),
            ('proprioceptive-blocked',active_run,sham_run,0.),
            ('held-voltage',factory(held=True),factory(held=True,sham=True),0.)]:
        destination=out/name;destination.mkdir()
        s=sensory_offsets.copy()
        if name in ('chemical-blocked','proprioceptive-blocked'):
            params=apply_joint_offsets(initial,offsets,spec);gains=np.exp(s)
            if name=='chemical-blocked':gains[0]=0.
            else:gains[1:]=0.
            a=run(params,times,gains);b=reference(params,times,gains)
            metrics=response_metrics(a,b,model.neuron_ids,protocol)
            result=dict(metrics=metrics);traces=dict(active=a,sham=b)
        else:result,traces=evaluate(offsets,s,run,reference,shift)
        for label,trace in traces.items():np.savez_compressed(destination/(label+'-trajectory.npz'),**trace)
        write_json(destination/'metrics.json',result)
        a=traces['active']
        if name=='active':
            write_json(destination/'spontaneous_sham_signal_chain.json',signal_chain(traces['sham'],args.dt))
            coarse_difference=a['dense_joints']-traces['sham']['dense_joints']
            coarse_time=a['dense_time_s']
        if name=='refined':
            difference=a['dense_joints']-traces['sham']['dense_joints']
            aligned=np.column_stack([np.interp(coarse_time,a['dense_time_s'],difference[:,j]) for j in range(11)])
            error=float(np.max(np.abs(aligned-coarse_difference)))
            refinement=dict(maximum_cue_effect_difference_rad=error,
                allowed_difference_rad=.25*protocol.bend_target_rad,
                passed=bool(error<=.25*protocol.bend_target_rad),
                scope='Engineering convergence of cue-minus-sham body response, not biological or voltage error bound')
        write_motion_preview(destination/'motion.html',a['dense_joints'],a['dense_position'],a['dense_heading'],
            dt=args.dt/2 if name=='refined' else args.dt,body_length=.001,seconds=protocol.duration_s,
            title='E055 '+name+'; sensory-evoked response, not autonomous skill')
        validation.append(dict(case=name,metrics=result['metrics']))
        print(json.dumps(dict(validation=validation[-1])),flush=True)
    rows={r['case']:r['metrics'] for r in validation}
    controls=all(rows[n]['cue_specific_peak_rad']<.1*protocol.bend_target_rad
        and rows[n]['awa_depolarization_mV']<.1*protocol.sensory_target_mV for n in ('chemical-blocked','held-voltage'))
    passed=bool(all(rows[n]['passed'] for n in ('active','dose-low','dose-high','onset-shift','refined'))
        and controls and refinement['passed'])
    summary=dict(experiment='E055',stage='contract_relax',status='completed',passed=passed,
        baseline=baseline,best=best,iterations=completed,selected_iteration=selected_iteration,
        initial_iteration=initial_iteration,last_iteration=initial_iteration+completed,
        validation=validation,blocked_controls_passed=bool(controls),elapsed_seconds=time.monotonic()-started,
        refinement_comparison=refinement,
        stop_reason='time_budget' if completed<args.iterations else 'iterations',
        checkpoint_sha256=file_sha256(out/'network_best.npz'),parent_checkpoint_sha256=parent_summary['checkpoint_sha256'],
        original_anchor_sha256=cfg['network_sha256'],frozen_fields_unchanged=True,
        neuron_connection_parameters_changed_from_parent=int(np.count_nonzero(offsets-start_offsets)),
        neuron_connection_parameters_changed_this_run=int(np.count_nonzero(offsets-run_start_offsets)),
        resume_checkpoint_sha256=resume_summary['checkpoint_sha256'] if resume_summary else None,
        sensory_parameters_sha256=file_sha256(out/'sensory_parameters.json'),
        sensory_gain_multipliers=np.exp(sensory_offsets).tolist(),biological_fidelity_qualified=False,
        autonomous_oscillation_qualified=False,next_stage_launched=False)
    if file_sha256(source/'network_best.npz')!=parent_summary['checkpoint_sha256']:raise ValueError('Parent changed')
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,allow_nan=False),flush=True)
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
