"""Probe bounded posterior channel regimes without injected current or training."""
import argparse
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from r_elegans.brain.body_amplitude import sustained_excursions
from r_elegans.brain.conductance_network import load_network,network_recording,save_network
from r_elegans.brain.joint_autonomous_fit import joint_fit_reference,apply_joint_offsets
from r_elegans.brain.posterior_excitability import density_profiles,isolated_recording
from r_elegans.brain.tail_amplitude import tail_focus_cells
from r_elegans.body.neuromuscular import build_muscle_projection
from r_elegans.model import load_builtin_model
from scripts.autonomous_warm_start import copy_warm_start,read_warm_start
from scripts.experiment_records import capture_run_record,file_sha256


def summarize(voltage,calcium,threshold,slope,ids):
    v=np.asarray(voltage);ca=np.asarray(calcium)
    if not np.isfinite(v).all() or not np.isfinite(ca).all():
        return dict(valid=False,rhythmic_count=0,reason='nonfinite')
    amplitude,_=sustained_excursions(v)
    release=expit((v-np.asarray(threshold))/np.asarray(slope))
    release_amp,_=sustained_excursions(release)
    late=v[len(v)//2:]
    valid=np.all((v>-120)&(v<80),axis=0)&(ca>=0).all(axis=0)
    qualified=valid&(amplitude>=.05)
    return dict(valid=bool(valid.all()),rhythmic_count=int(qualified.sum()),
        rhythmic_neurons=[ids[i] for i in np.flatnonzero(qualified)],
        maximum_sustained_p2p_mV=float(amplitude.max()),median_sustained_p2p_mV=float(np.median(amplitude)),
        cells=[dict(neuron=name,valid=bool(valid[i]),rhythmic=bool(qualified[i]),
            sustained_p2p_mV=float(amplitude[i]),sustained_release_p2p=float(release_amp[i]),
            late_mean_voltage_mV=float(late[:,i].mean()),late_raw_p2p_mV=float(np.ptp(late[:,i]))) for i,name in enumerate(ids)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--random-profiles',type=int,default=8)
    parser.add_argument('--seed',type=int,default=4000)
    parser.add_argument('--screen-seconds',type=float,default=12.)
    parser.add_argument('--confirm-seconds',type=float,default=30.)
    args=parser.parse_args()
    if args.random_profiles<0 or args.seed<0 or not all(math.isfinite(x) and x>=12 for x in (args.screen_seconds,args.confirm_seconds)):
        parser.error('Nonnegative seed/count and finite durations>=12 s required')
    if args.confirm_seconds<args.screen_seconds:parser.error('Confirmation must be at least screen duration')
    repo=Path(__file__).resolve().parents[1];out=args.output_dir.expanduser().resolve();source=args.worker.expanduser().resolve()
    if out==repo or repo in out.parents or out==source or source in out.parents:parser.error('Output must be outside repository and parent worker')
    warm=read_warm_start(source)
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,args)
    copy_warm_start(source,out/'parent_worker')
    parent=out/'parent_worker';original,fixed,config=load_network(parent/'source_inputs/network.npz')
    current,current_fixed,current_config=load_network(parent/'network_best.npz')
    fixed,spec=joint_fit_reference(fixed);offset=np.asarray(warm['state']['best'],float)
    rebuilt=apply_joint_offsets(original,jnp.asarray(offset,dtype=original.log_channel_scale.dtype),spec)
    if current_config!=config or not all(np.array_equal(a,b) for a,b in zip(jax.tree.leaves((rebuilt,fixed)),jax.tree.leaves((current,current_fixed)))):
        raise ValueError('Original offsets fail exact parent reconstruction')
    if not np.all(np.asarray(fixed.bias_pA)==0):raise ValueError('Zero external current required')
    model=load_builtin_model();focus=tail_focus_cells(model.neuromuscular_params.synapse_weights,build_muscle_projection(11,head_at_end=True))
    ids=[model.neuron_ids[i] for i in focus]
    channel_offsets=np.zeros_like(np.asarray(original.log_channel_scale),dtype=float)
    a=spec.counts['channel_densities'];channel_offsets[spec.channels]=offset[:a]
    profiles,rng_state=density_profiles(channel_offsets[focus],spec.channels[focus],warm['maximum_factor'],args.random_profiles,args.seed)
    np.savez_compressed(out/'profiles.npz',focus_indices=focus,neuron_ids=np.asarray(ids),
        names=np.asarray([name for name,_ in profiles]),original_anchor_offsets=np.stack([value for _,value in profiles]))
    (out/'rng_state.json').write_text(json.dumps(rng_state,indent=2)+'\n')
    dt=.002
    cells=jax.tree.map(lambda x:x[focus],fixed.cells)
    anchor=np.asarray(original.log_channel_scale)[focus]
    initial=fixed.initial_voltage_mV[focus]
    threshold,slope=fixed.release_threshold_mV[focus],fixed.release_slope_mV[focus]
    kernels={};start=time.monotonic()
    def isolated(name,values,seconds,perturbation=0.,substeps=20):
        key=(seconds,substeps)
        if key not in kernels:
            kernels[key]=jax.jit(lambda conductance,voltage:isolated_recording(
                cells._replace(conductances_nS=conductance),voltage,dt,round(seconds/dt),substeps))
        g=cells.conductances_nS*jnp.exp(jnp.asarray(anchor+values,dtype=initial.dtype))
        v,ca=jax.device_get(kernels[key](g,initial+perturbation))
        metrics=summarize(v,ca,threshold,slope,ids)
        np.savez_compressed(out/(name+'.npz'),voltage_mV=v,calcium_uM=ca,neuron_ids=np.asarray(ids))
        row=dict(case=name,seconds=seconds,step_ms=dt*1000/substeps,initial_perturbation_mV=perturbation,metrics=metrics)
        print(json.dumps(dict(case=name,rhythmic_count=metrics['rhythmic_count'],maximum_sustained_p2p_mV=metrics.get('maximum_sustained_p2p_mV'),elapsed_seconds=time.monotonic()-start)),flush=True)
        return row
    results=[]
    for name,values in profiles:
        results.append(isolated('screen-'+name,values,args.screen_seconds))
        (out/'screen-progress.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    rank=lambda i:(results[i]['metrics']['rhythmic_count'],results[i]['metrics'].get('maximum_sustained_p2p_mV',0.))
    selected=max(range(len(profiles)),key=rank)
    reason='Highest rhythmic cell count, then largest sustained excursion; no feasibility proof from negative screen'
    if rank(selected)==(0,0.):
        selected=[n for n,_ in profiles].index('inward_high_outward_low')
        reason='All screened sustained excursions zero; predefined strong-inward probe selected as a diagnostic, not a successful regime'
    name,values=profiles[selected]
    confirmations=[isolated('confirm-long',values,args.confirm_seconds),
        isolated('confirm-plus',values,args.confirm_seconds,1.),
        isolated('confirm-minus',values,args.confirm_seconds,-1.),
        isolated('confirm-refined',values,args.confirm_seconds,substeps=40)]
    robust=set(confirmations[0]['metrics'].get('rhythmic_neurons',[]))
    for row in confirmations[1:]:robust&=set(row['metrics'].get('rhythmic_neurons',[]))
    # Density-only intervention on the selected posterior cells. Keep learned edges.
    scales=current.log_channel_scale.at[focus].set(jnp.asarray(anchor+values,dtype=current.log_channel_scale.dtype))
    candidate=current._replace(log_channel_scale=scales)
    save_network(out/'diagnostic_candidate.npz',candidate,fixed,config)
    connected=[];frames=round(args.confirm_seconds/dt)
    net=jax.jit(lambda p,f:network_recording(p,f,dt,frames,20,config))
    keep=np.zeros(302,bool);keep[focus]=True;within=jnp.asarray(keep[:,None]&keep[None,:])
    subcircuit=fixed._replace(chemical_mask=fixed.chemical_mask*within,gap_mask=fixed.gap_mask*within)
    for label,p,f in [('current-full',current,fixed),('current-posterior-subcircuit',current,subcircuit),('candidate-full',candidate,fixed)]:
        trace=jax.device_get(net(p,f));v=np.asarray(trace.voltage_mV);ca=np.asarray(trace.calcium_uM)
        posterior=summarize(v[:,focus],ca[:,focus],threshold,slope,ids)
        all_metrics=summarize(v,ca,f.release_threshold_mV,f.release_slope_mV,list(model.neuron_ids))
        np.savez_compressed(out/(label+'.npz'),voltage_mV=v,calcium_uM=ca,neuron_ids=np.asarray(model.neuron_ids))
        connected.append(dict(case=label,posterior=posterior,all_neurons=all_metrics))
        print(json.dumps(dict(case=label,posterior_rhythmic_count=posterior['rhythmic_count'],all_rhythmic_count=all_metrics['rhythmic_count'])),flush=True)
    summary=dict(experiment='E040',status='completed',source_parent=str(source),parent_essential_sha256=warm['essential_sha256'],
        original_anchor_sha256=file_sha256(parent/'source_inputs/network.npz'),profiles_sha256=file_sha256(out/'profiles.npz'),
        maximum_factor=warm['maximum_factor'],injected_current_pA=0.,focus_neurons=ids,screen=results,selected_profile=name,
        selection_reason=reason,confirmation=confirmations,robust_isolated_rhythmic_neurons=sorted(robust),connected=connected,
        elapsed_seconds=time.monotonic()-start,interpretation='Finite bounded zero-input channel feasibility screen; diagnostic graph ablations, no behavior qualification',
        limitations=['Independent-cell oscillation is not necessary for emergent network rhythms.',
                     'Negative finite screen does not establish impossibility within bounds.',
                     'No body simulated;network candidate is not promoted to fitted inputs.',
                     'Subcircuit removes all connections to outside neurons;that is a diagnostic graph intervention.',
                     'Connected comparisons at0.1ms only;isolated selected regime has timestep and initial-voltage checks.',
                     'Confirmation amplitude/period agreement is reported for inspection, not certified by shared rhythm labels.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(selected_profile=name,robust_isolated_rhythms=len(robust),elapsed_seconds=summary['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
