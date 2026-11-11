"""E054 read-only sustained dynamics, signal-chain and original-bound inspection."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

from scripts.experiment_records import capture_run_record,file_sha256
from scripts.assess_wet_surface import verify_reference
from scripts.paired_resume import read_resume


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--duration',type=float,default=40.)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--brain-step-ms',type=float,default=.1)
    args=p.parse_args()
    import math
    if not all(math.isfinite(v) and v>0 for v in (args.duration,args.dt,args.brain_step_ms)) or not 1<=args.duration<=600:
        p.error('Duration1..600s and positive integration settings required')
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION','highest')
    import numpy as np
    import jax
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets
    from r_elegans.brain.paired_fitting import make_rollout
    from r_elegans.rl.full_network_actor import actor_from_arrays
    from r_elegans.model import load_builtin_model
    from r_elegans.data.paths import get_data_root
    from r_elegans.paired_diagnostics import bound_occupancy,signal_chain
    from r_elegans.full_body_assessment import write_motion_preview
    source=args.run.resolve()
    cfg=json.loads((source/'configuration.json').read_text())
    _,anchor,readout=verify_reference(source/'inputs/parent_lineage')
    initial,original_fixed,network_config=load_network(anchor)
    fixed,spec=joint_fit_reference(original_fixed)
    offsets,iteration,_,parent_summary=read_resume(source,cfg,spec.size)
    params,actual_fixed,actual_config=load_network(source/'network_best.npz')
    expected=apply_joint_offsets(initial,offsets,spec)
    if (actual_config!=network_config or asdict(network_config)!=cfg['network_config'] or
        file_sha256(readout)!=cfg['readout_sha256'] or file_sha256(anchor)!=cfg['network_sha256'] or
        any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(
            jax.tree_util.tree_leaves((expected,fixed)),jax.tree_util.tree_leaves((params,actual_fixed))))):
        raise ValueError('Saved checkpoint, original anchor/offsets and fixed fields disagree')
    manifest=json.loads((source/'inputs/manifest.json').read_text())
    asset=Path(__file__).resolve().parents[1]/'r_elegans/assets/runtime_model_v1.json'
    if file_sha256(asset)!=manifest['canonical_model_asset_sha256']:raise ValueError('Runtime asset changed')
    if np.any(np.asarray(fixed.bias_pA)!=0):raise ValueError('Zero-bias parent required')
    out=get_data_root(args.output_dir,must_exist=False)
    if out==source or source in out.parents:p.error('Output must be outside parent run')
    out.mkdir(parents=True,exist_ok=False)
    capture_run_record(out,Path(__file__).resolve().parents[1],args)
    if any(p.is_symlink() for p in source.rglob('*')):raise ValueError('Parent archive contains symlinks')
    shutil.copytree(source,out/'parent_run')
    if file_sha256(out/'parent_run/network_best.npz')!=parent_summary['checkpoint_sha256']:
        raise ValueError('Copied parent checkpoint mismatch')
    bounds=bound_occupancy(offsets,spec.counts,cfg['arguments']['maximum_factor'])
    write(out/'bounds.json',bounds)
    print(json.dumps(dict(bounds=bounds)),flush=True)
    with np.load(readout,allow_pickle=False) as z:parent=actor_from_arrays(z)
    model=load_builtin_model()
    medium=cfg['arguments']['media']
    if medium!=['wet-agar']:raise ValueError('This initial diagnostic targets the wet-agar paired search')
    cases=[('active',args.dt,args.brain_step_ms,False),
           ('refined',args.dt/2,args.brain_step_ms/2,False),
           ('held-voltage',args.dt,args.brain_step_ms,True)]
    results=[]
    times=np.linspace(0,args.duration,max(2,math.ceil(args.duration/.01)+1))
    for name,dt,brain_step,held in cases:
        destination=out/name;destination.mkdir()
        print(json.dumps(dict(starting=name,simulation_seconds=args.duration,dt=dt,held_voltage=held)),flush=True)
        run=make_rollout(fixed,network_config,parent,model,medium='wet-agar',dt=dt,
            brain_step_ms=brain_step,burn_in=0.,sampling='point',record_dense=True,hold_voltage=held)
        trace=run(params,times)
        np.savez_compressed(destination/'trajectory.npz',**trace)
        report=signal_chain(trace,dt)
        write(destination/'metrics.json',report)
        late=len(trace['dense_time_s'])//2
        write_motion_preview(destination/'motion.html',trace['dense_joints'][late:],trace['dense_position'][late:],
            trace['dense_heading'][late:],dt=dt,body_length=.001,seconds=min(10,args.duration/2),
            title='E054 '+name+'; late half from '+str(float(trace['dense_time_s'][late]))+'s, no training')
        results.append(dict(case=name,dt_s=dt,metrics=report))
        print(json.dumps(dict(finished=name,repeated_joints=report['dense_joints']['repeated_channel_count'],
            repeated_muscles=report['dense_muscles']['repeated_channel_count'],
            repeated_voltage_neurons=[model.neuron_ids[i] for i in report['dense_voltage_mV']['repeated_indices']],
            maximum_sustained_voltage_p2p_mV=max(report['dense_voltage_mV']['sustained_p2p']),
            maximum_sustained_nmj_activation_p2p=max(report['dense_muscles']['sustained_p2p']),
            maximum_sustained_command_p2p=max(report['dense_command']['sustained_p2p']),
            maximum_sustained_filtered_drive_p2p=max(report['dense_filtered_drive']['sustained_p2p']),
            nmj_near_extremes_fraction=report['nmj_activation_near_extremes_fraction'],
            maximum_sustained_joint_p2p_rad=max(report['dense_joints']['sustained_p2p']),
            late_shape_pc1_fraction=report['late_shape_pc1_fraction'])),flush=True)
    comparison={}
    with np.load(out/'active/trajectory.npz',allow_pickle=False) as coarse,np.load(out/'refined/trajectory.npz',allow_pickle=False) as fine:
        for key in ('dense_joints','dense_voltage_mV','dense_filtered_drive'):
            a,b=coarse[key],fine[key][::2]
            if a.shape!=b.shape:raise ValueError('Refined/coarse grids fail to align')
            comparison[key+'_max_abs_difference']=float(np.max(np.abs(a-b)))
            comparison[key+'_late_max_abs_difference']=float(np.max(np.abs(a[len(a)//2:]-b[len(b)//2:])))
    report=dict(experiment='E054',status='completed',source_run=str(source),
        checkpoint_sha256=parent_summary['checkpoint_sha256'],global_iteration=iteration,bounds=bounds,cases=results,
        simulation_seconds=args.duration,timestep_comparison=comparison,parameters_changed=False,external_drive=False,
        scope='Dense late-half repeated excursions; no calibrated mounted-medium or biological pass',
        late_start_seconds=float(args.duration/2),
        medium='Unchanged1.7% wet-agar proxy; not calibrated2% coverslip conditions')
    if file_sha256(source/'network_best.npz')!=parent_summary['checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed during assessment')
    write(out/'summary.json',report)
    print('Diagnostic complete: '+str(out/'summary.json'),flush=True)


if __name__=='__main__':main()
