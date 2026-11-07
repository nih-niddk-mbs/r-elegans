"""Search a published conductance-model regime for persistent unpaced activity.

This tests a qualitative RMD-model dynamical regime, NOT a fitted DA/VA
pacemaker. No periodic input or phase variable. Isolated cells have zero bias;
connected tests zero RMD bias and retain the other cells' constant parent bias.
"""
import argparse
import json
import math
from dataclasses import replace
import hashlib
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain.conductance_network import network_recording,load_network,save_network
from r_elegans.brain.rhythm_qualification import isolated_rmd_grid,rhythm_metrics,apply_rmd_rhythm_prior,compare_cycles
from r_elegans.brain.single_compartment import CHANNEL_INDEX
from r_elegans.model import load_builtin_model
if __package__:
    from .experiment_records import capture_run_record
else:
    from experiment_records import capture_run_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--duration',type=float,default=10.)
    parser.add_argument('--initial-voltage',type=float,default=-35.)
    parser.add_argument('--max-step-ms',type=float,default=.05)
    parser.add_argument('--confirm',action='store_true',help='One regime and causal/initial-state controls')
    parser.add_argument('--initial-network',type=Path,help='Enable full anatomical-network assay')
    parser.add_argument('--coupling-scale',type=float,default=1.)
    parser.add_argument('--remove-rmd-cca1',action='store_true')
    args=parser.parse_args()
    if not all(math.isfinite(x) for x in (args.duration,args.initial_voltage,args.max_step_ms)) or args.duration<10 or args.max_step_ms<=0:
        parser.error('Finite settings, duration >=10 s and positive timestep required')
    repo=Path(__file__).resolve().parents[1];output=args.output_dir.expanduser().resolve()
    if output==repo or repo in output.parents:parser.error('Keep artifacts external')
    output.mkdir(parents=True,exist_ok=False);capture_run_record(output,repo,args)
    if args.coupling_scale<=0 or not math.isfinite(args.coupling_scale):parser.error('Positive coupling scale required')
    if args.initial_network and args.confirm:parser.error('Choose isolated confirmation or connected-network assay')
    if args.remove_rmd_cca1 and not args.initial_network:parser.error('RMD ablation requires --initial-network')
    leaks=[.03,.045,.06,.075,.09,.105,.12]
    # Fig.11's CCA1 axis extends to ~70 nS; the ordinary 3.1 nS RMD
    # preset is outside most of its periodic region. Keep low-g controls.
    cca=[.3,3.1,10.,20.,30.,50.,65.]
    params,fixed,pairs=isolated_rmd_grid(leaks,cca,initial_voltage=args.initial_voltage)
    dt=.002;substeps=math.ceil(dt*1000/args.max_step_ms);frames=math.ceil(args.duration/dt)
    config=None;model=load_builtin_model()
    if args.confirm:
        params,fixed,pairs=isolated_rmd_grid([.045],[30.,30.,30.,0.,3.1],initial_voltage=args.initial_voltage)
        # Candidate, alternate voltage basins, CCA ablation, stable ordinary RMD.
        fixed=fixed._replace(initial_voltage_mV=jnp.array([-35.,-34.,-70.,-35.,-35.]),
            cells=fixed.cells._replace(conductances_nS=fixed.cells.conductances_nS.at[4,CHANNEL_INDEX['LEAK']].set(.4)))
        pairs[4,0]=.4
    elif args.initial_network:
        params,fixed,config=load_network(args.initial_network)
        params,fixed,indices=apply_rmd_rhythm_prior(params,fixed,model.neuron_ids,initial_voltage=args.initial_voltage)
        if args.remove_rmd_cca1:
            fixed=fixed._replace(cells=fixed.cells._replace(conductances_nS=fixed.cells.conductances_nS.at[indices,CHANNEL_INDEX['CCA1']].set(0.)))
        config=replace(config,synaptic_unit_nS=config.synaptic_unit_nS*args.coupling_scale)
        shutil.copy2(args.initial_network,output/'parent_network.npz')
        save_network(output/'network_candidate.npz',params,fixed,config)
    run=lambda n:network_recording(params,fixed,dt,frames,n) if config is None else network_recording(params,fixed,dt,frames,n,config)
    trace=jax.jit(lambda:run(substeps))()
    v=np.asarray(trace.voltage_mV)
    if args.initial_network:
        results=[dict(neuron=model.neuron_ids[i],metrics=rhythm_metrics(v[:,i],dt)) for i in range(302)]
    else:
        results=[dict(leak_nS=float(pair[0]),cca1_nS=float(pair[1]),
            metrics=rhythm_metrics(v[:,i],dt)) for i,pair in enumerate(pairs)]
    summary=dict(experiment='E027',scope='isolated RMD conductance-model regime, not calibrated motor physiology',
        duration_s=frames*dt,sample_dt_s=dt,integration_step_ms=dt*1000/substeps,
        initial_voltage_mV=args.initial_voltage,applied_current_pA=0.,
        passed_candidates=sum(r['metrics']['passed'] for r in results),results=results,
        calcium_finite=bool(np.isfinite(trace.calcium_uM).all()),
        calcium_range_uM=[float(trace.calcium_uM.min()),float(trace.calcium_uM.max())])
    if args.confirm or args.initial_network:
        refined=jax.jit(lambda:run(substeps*2))()
        summary['substep_refinement']={
            'maximum_voltage_difference_mV':float(np.max(np.abs(np.asarray(refined.voltage_mV)-v))),
            'voltage_rmse_mV':float(np.sqrt(np.mean((np.asarray(refined.voltage_mV)-v)**2))),
            'cycle_comparisons':[compare_cycles(v[:,i],np.asarray(refined.voltage_mV)[:,i],dt) for i in range(v.shape[1])],
            'refined_results':[rhythm_metrics(np.asarray(refined.voltage_mV)[:,i],dt) for i in range(v.shape[1])]}
        np.savez_compressed(output/'refined.npz',voltage_mV=np.asarray(refined.voltage_mV),calcium_uM=np.asarray(refined.calcium_uM))
    if args.confirm:
        summary['mode']='isolated_confirmation'
        summary['conditions']=['candidate','initial_minus34','initial_minus70','CCA1_removed','ordinary_RMD_control']
        summary['initial_voltages_mV']=np.asarray(fixed.initial_voltage_mV).tolist()
    if args.initial_network:
        summary.update(mode='connected_network',scope='full 302-cell channel network; exploratory RMD prior',
            coupling_scale=args.coupling_scale,synaptic_unit_nS=config.synaptic_unit_nS,
            parent_sha256=hashlib.sha256(args.initial_network.read_bytes()).hexdigest(),
            rmd_cca1_removed=args.remove_rmd_cca1,
            fixed_bias_policy='RMD cells zero; other cells retain the parent constant intrinsic-current compensation',
            rmd_neurons=[model.neuron_ids[int(i)] for i in indices])
    np.savez_compressed(output/'traces.npz',voltage_mV=v,calcium_uM=np.asarray(trace.calcium_uM),
        pairs=pairs if not args.initial_network else np.empty((0,2)),
        neuron_ids=np.asarray(model.neuron_ids) if args.initial_network else np.asarray([],dtype='U1'))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed_candidates=summary['passed_candidates'],calcium_finite=summary['calcium_finite'],
        candidates=[r for r in results if r['metrics']['passed']])),flush=True)


if __name__=='__main__':main()
