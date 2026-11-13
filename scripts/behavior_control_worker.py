"""E072 stop/restart learning; E073 propulsion preserves the same control gates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np

from scripts.experiment_records import capture_run_record,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--mode',choices=['fit','validate'],required=True)
    p.add_argument('--stage',choices=['control','swim'],required=True)
    p.add_argument('--until',type=float,required=True)
    p.add_argument('--seed',type=int,default=7200)
    p.add_argument('--case',default='nominal')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--coordinates',type=Path)
    p.add_argument('--stop-flag',type=Path)
    a=p.parse_args();root=a.inputs.resolve();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    if out==repo or repo in out.parents or out==root or root in out.parents:p.error('Separate external output required')
    from scripts.neural_timing_setup import setup
    from scripts.compare_rhythm_worker import verify_inputs
    from scripts.assess_wet_surface import write_json
    from r_elegans.brain.conductance_network import load_network,save_network
    from r_elegans.behavior_control import (ControlProtocol,bath_schedule,metrics,case_protocol,CONTROL_GROUPS,SENSORY_CELLS)
    from r_elegans.neural_timing import propose_timing
    from r_elegans.wall_budget import check_deadline,WallDeadline
    check_deadline(a.until);d=setup(root);ids=d['model'].neuron_ids
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,a)
    body=d['liquid_body'];control_cells={ids.index(n) for group in CONTROL_GROUPS.values() for n in group}
    sensory_cells={ids.index(n) for n in SENSORY_CELLS}
    ci=np.nonzero(d['spec'].channels);ei=np.nonzero(d['spec'].chemical);gi=np.nonzero(np.triu(d['spec'].gap,1))
    allowed=np.r_[np.isin(ci[0],list(control_cells|sensory_cells)),
                  np.isin(ei[0],list(control_cells))|np.isin(ei[1],list(control_cells)),
                  np.isin(gi[0],list(control_cells))|np.isin(gi[1],list(control_cells))]
    if a.stage=='swim':allowed[:]=True
    np.savez_compressed(out/'parameter_origin.npz',origin=d['origin'],lower=d['lower'],upper=d['upper'],trainable=allowed)
    rollouts={}
    def run(params,protocol,case='nominal'):
        key=(protocol,case)
        if key in rollouts:
            rollout,gains=rollouts[key]
            return rollout(params,np.linspace(0,protocol.duration,121),gains)
        dt=.001 if case=='body-half' else .002
        brain=.05 if case=='neural-half' else .1
        fixed=None
        if case.startswith('controller-'):
            import jax.numpy as jnp
            selected=[ids.index(n) for n in CONTROL_GROUPS[case.removeprefix('controller-')]]
            mask=np.asarray(d['fixed'].chemical_mask).copy();mask[:,selected]=False
            gap=np.asarray(d['fixed'].gap_mask).copy();gap[:,selected]=False;gap[selected,:]=False
            fixed=d['fixed']._replace(chemical_mask=jnp.asarray(mask),gap_mask=jnp.asarray(gap))
        gains=d['gains'].copy()
        if case=='chemical-blocked':gains[0]=0
        rollout=d['factory'](dt,brain,0.,case=='held-voltage',wall_deadline=a.until,
                             world_override=bath_schedule(protocol),fixed_override=fixed)
        rollouts[key]=(rollout,gains)
        return rollout(params,np.linspace(0,protocol.duration,121),gains)
    origin=d['origin'].copy();protocol=case_protocol(a.case) if a.mode=='validate' else ControlProtocol()
    write_json(out/'configuration.json',dict(experiment='E072' if a.stage=='control' else 'E073',arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        protocol=asdict(protocol),parent_sha256=d['manifest']['continuation_parent_sha256'],
        input_manifest_sha256=file_sha256(root/'manifest.json'),control_groups=CONTROL_GROUPS,
        trainable_coordinates=int(allowed.sum()),full_coordinate_count=len(origin),physical_clock_unchanged=True,
        fixed='Leak/capacitance/kinetics/reversals/release/topology, NMJ/muscle/water mechanics, sensory routing/kinetics/gains',
        cue_scope='Learned chemical cue association through AWA/ASE; not a claimed natural odor/salt stop reflex',
        parent_gate_required=a.stage=='swim',
        virtual_intervention=a.case if a.mode=='validate' and a.case.startswith('controller-') else None))
    if a.mode=='validate':
        if a.checkpoint is None or a.coordinates is None:p.error('Selected checkpoint and coordinates required')
        checkpoint_hash=file_sha256(a.checkpoint);coordinate_hash=file_sha256(a.coordinates)
        with np.load(a.coordinates,allow_pickle=False) as z:x=z['offsets']
        params=d['decode'](x);actual,fixed,config=load_network(a.checkpoint)
        if config!=d['config'] or any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(actual,params)):
            raise ValueError('Validation checkpoint/coordinate mismatch')
        import jax
        if any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree_util.tree_leaves(fixed),jax.tree_util.tree_leaves(d['fixed']))):
            raise ValueError('Validation fixed fields changed')
        protocol=case_protocol(a.case);trace=run(params,protocol,a.case)
        baseline=json.loads((root/'behavior_baseline.json').read_text())
        m=metrics(trace,protocol,baseline)
        active_windows,rest_windows=protocol.windows();neural={}
        for group,names in CONTROL_GROUPS.items():
            rows=[]
            for name in names:
                voltage=trace['dense_voltage_mV'][:,ids.index(name)]
                summaries={}
                for label,windows in [('active',active_windows),('rest',rest_windows)]:
                    summaries[label]=[dict(mean_mV=float(np.mean(voltage[(trace['dense_time_s']>=a)&(trace['dense_time_s']<b)])),
                        p2p_mV=float(np.ptp(voltage[(trace['dense_time_s']>=a)&(trace['dense_time_s']<b)]))) for a,b in windows]
                rows.append(dict(neuron=name,**summaries))
            neural[group]=rows
        m['descriptive_controller_voltage_profiles']=neural
        if checkpoint_hash!=file_sha256(a.checkpoint) or coordinate_hash!=file_sha256(a.coordinates):
            raise ValueError('Selected model changed during validation')
        np.savez_compressed(out/'trajectory.npz',**trace)
        np.savez_compressed(out/'coordinates.npz',offsets=x)
        result=dict(case=a.case,metrics=m,protocol=asdict(protocol),checkpoint_sha256=checkpoint_hash,
                    trajectory_sha256=file_sha256(out/'trajectory.npz'),simulation_seconds=protocol.duration)
        if a.case in ('nominal','repeat'):
            from r_elegans.full_body_assessment import write_motion_preview
            result['motion_preview']=write_motion_preview(out/'motion.html',trace['dense_joints'],trace['dense_position'],
                trace['dense_heading'],dt=.002,body_length=.001,
                seconds=protocol.duration,title=f'{a.stage}: {a.case}; saved trajectory, qualification reported separately')
        write_json(out/'summary.json',result);print(json.dumps(result),flush=True);verify_inputs(root);return
    baseline_kind=d['manifest'].get('reference_protocol','no-cue')
    initial_protocol=ControlProtocol(sham=True) if baseline_kind=='no-cue' else protocol
    initial=run(d['decode'](origin),initial_protocol)
    reference=root/'parent_training_trace.npz'
    # Validate every common recorded timestamp before fitting any parameter.
    with np.load(reference,allow_pickle=False) as z:
        use=initial['dense_time_s']<=z['dense_time_s'][-1]+1e-10
        if not np.array_equal(initial['dense_time_s'][use],z['dense_time_s']):raise ValueError('Parent reference time grids differ')
        errors={key:float(np.max(np.abs(initial[key][use]-z[key]))) for key in
                ('dense_voltage_mV','dense_calcium_uM','dense_muscles','dense_joints','dense_filtered_drive')}
        if any(v>1e-8 for v in errors.values()):raise ValueError('Rebuilt parent trajectory differs: '+str(errors))
    write_json(out/'source_reference_check.json',dict(passed=True,maximum_errors=errors))
    baseline=metrics(initial,initial_protocol)
    write_json(out/'baseline.json',baseline)
    if not baseline['initial_movement_passed']:raise ValueError('Parent initial movement is not qualified')
    if a.stage=='swim':
        from r_elegans.behavior_control import qualify_behavior
        gate=json.loads((root/'control_gate.json').read_text())
        if not qualify_behavior(d['manifest']['continuation_parent_sha256'],True,gate):
            raise ValueError('Swimming requires the exact validated control parent')
    x=origin.copy();rng=np.random.default_rng(a.seed);epoch='stop' if a.stage=='control' else 'swim'
    trace=run(d['decode'](x),protocol);m=metrics(trace,protocol,baseline)
    def loss(value,coords):
        data=value['stop_loss'] if epoch=='stop' else (value['restart_loss'] if epoch=='restart' else value['swim_loss'])
        return float(data+.001*np.mean((coords-origin)**2))
    best=dict(loss=loss(m,x),metrics=m);selected='initial';iterations=0
    def save(tag):
        path=out/'selected'/tag;path.mkdir(parents=True)
        save_network(path/'network.npz',d['decode'](x),d['fixed'],d['config'])
        np.savez_compressed(path/'coordinates.npz',offsets=x,epoch=epoch,iteration=iterations)
        np.savez_compressed(path/'trajectory.npz',**trace)
        write_json(path/'metrics.json',best);write_json(path/'rng_state.json',rng.bit_generator.state)
        for name,target in [('network.npz','network_best.npz'),('coordinates.npz','best_state.npz'),('trajectory.npz','best_training_trace.npz')]:shutil.copy2(path/name,out/target)
        write_json(out/'best.json',dict(selected=tag,result=best))
    save(selected)
    with (out/'history.jsonl').open('w') as history:
        while time.monotonic()<a.until-30:
            if a.stop_flag and a.stop_flag.exists():break
            if epoch=='stop' and all(r['passed'] for r in m['off']):
                epoch='restart';best=dict(loss=loss(m,x),metrics=m);selected=f'epoch-restart-{iterations:06d}';save(selected)
            complete=m['control_passed'] if a.stage=='control' else m['swimming_passed']
            if complete:break
            focus=sorted(control_cells|sensory_cells)
            if a.stage=='swim':
                import re
                focus=[i for i,n in enumerate(ids) if re.fullmatch(r'(DA|DB|DD|VA|VB|VD|AS)\d+',n) or n.startswith(('RMD','SMD','SMB'))]
            candidate,proposal=propose_timing(x,d['spec'],rng,.05,d['lower'],d['upper'],focus)
            candidate[~allowed]=x[~allowed]
            accepted=False;error=None
            try:
                candidate_trace=run(d['decode'](candidate),protocol);value=metrics(candidate_trace,protocol,baseline)
                permitted=value['safety_passed'] and value['initial_movement_passed']
                if epoch=='restart':permitted=permitted and all(r['passed'] for r in value['off'])
                if epoch=='swim':permitted=permitted and value['control_passed']
                objective=loss(value,candidate)
                if permitted and objective<best['loss']:
                    x=candidate;trace=candidate_trace;m=value;best=dict(loss=objective,metrics=m);accepted=True
            except WallDeadline:break
            except (ValueError,RuntimeError,FloatingPointError) as e:error=str(e)
            iterations+=1
            if accepted:selected=f'iteration-{iterations:06d}';save(selected)
            row=dict(iteration=iterations,epoch=epoch,accepted=accepted,proposal=proposal,error=error,best=best,rng_after=rng.bit_generator.state)
            history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
            np.savez_compressed(out/'optimizer_state.npz',offsets=x,epoch=epoch,iteration=iterations)
            write_json(out/'rng_state.json',rng.bit_generator.state);write_json(out/'progress.json',row)
    if not np.array_equal(x[~allowed],origin[~allowed]):raise ValueError('Frozen control-stage coordinates changed')
    save_network(out/'network_final.npz',d['decode'](x),d['fixed'],d['config'])
    np.savez_compressed(out/'optimizer_state.npz',offsets=x,epoch=epoch,iteration=iterations)
    write_json(out/'rng_state.json',rng.bit_generator.state)
    verify_inputs(root)
    result=dict(stage=a.stage,completed=True,iterations=iterations,epoch=epoch,best=best,
                nominal_passed=m['control_passed'] if a.stage=='control' else m['swimming_passed'],
                checkpoint_sha256=file_sha256(out/'network_best.npz'),selected=selected,
                coordinates_changed=int(np.count_nonzero(x-origin)),frozen_coordinates_unchanged=True)
    write_json(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
