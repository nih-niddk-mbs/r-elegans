"""E050 incremental joint calcium/movement statistics fit, not a biological fit.

Uses only frozen E049 training folds. Shared unknown channel densities and
existing synapse strengths; fixed reporter/NMJ/mechanics. No target current,
periodic drive, sensory teacher or source bearing. Mechanical mismatch and
unknown animal initial states prevent a source-condition fidelity claim.
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
from scripts.assess_wet_surface import verify_reference


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-data',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--uids',nargs='+',default=['2022-06-14-01'])
    p.add_argument('--frames',type=int,default=8)
    p.add_argument('--start-frame',type=int,default=0)
    p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--minutes',type=float,default=10.)
    p.add_argument('--seed',type=int,default=50)
    p.add_argument('--media',nargs='+',choices=['liquid','wet-agar'],default=['wet-agar'])
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--brain-step-ms',type=float,default=.1)
    p.add_argument('--burn-in',type=float,default=2.)
    p.add_argument('--maximum-factor',type=float,default=1.2)
    p.add_argument('--sigma',type=float,default=.03)
    p.add_argument('--basal-brightness',type=float,default=.1)
    p.add_argument('--sampling',choices=['point','boxcar'],default='boxcar')
    p.add_argument('--refine',action='store_true')
    args=p.parse_args()
    if (args.iterations<1 or len(set(args.media))!=len(args.media) or
        not all(math.isfinite(v) and v>0 for v in (args.minutes,args.sigma,args.basal_brightness)) or
        not math.isfinite(args.maximum_factor) or args.maximum_factor<=1):
        p.error('Positive settings, factor>1 and unique media required')
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION','highest')
    import numpy as np
    from r_elegans.brain.paired_fitting import training_windows,make_rollout
    from r_elegans.brain.conductance_network import load_network,save_network
    from r_elegans.brain.joint_autonomous_fit import (joint_fit_reference,apply_joint_offsets,
        joint_proposal,frozen_joint_fields_unchanged)
    from r_elegans.data.paired_observations import paired_objective,SOURCE_STEP_M,SOURCE_POINTS
    from r_elegans.data.paths import get_data_root
    from r_elegans.rl.full_network_actor import actor_from_arrays
    from r_elegans.model import load_builtin_model
    windows,manifest=training_windows(args.prepared_data,args.uids,args.frames,args.start_frame)
    expected=json.loads((Path(__file__).resolve().parents[1]/'experiments/records/E049-paired-data-admission.json').read_text())
    if manifest['split_sha256']!=expected['split_sha256'] or manifest['metadata_sha256']!=expected['metadata_sha256']:
        raise ValueError('Fit requires the frozen E049 cohort, not redrawn folds')
    source,network,readout=verify_reference(args.reference)
    out=get_data_root(args.output_dir,must_exist=False)
    if out==source or source in out.parents or out==args.prepared_data.resolve() or args.prepared_data.resolve() in out.parents:
        p.error('Do not write into parent or prepared data')
    out.mkdir(parents=True,exist_ok=False)
    capture_run_record(out,Path(__file__).resolve().parents[1],args)
    inputs=out/'inputs';inputs.mkdir()
    for path in (network,readout,args.prepared_data/'manifest.json',args.prepared_data/'splits.json'):
        shutil.copy2(path,inputs/path.name)
    # Preserve recursive history as well as self-contained inference inputs.
    if (source/'network_best.npz').is_file():
        from scripts.autonomous_warm_start import copy_warm_start
        copy_warm_start(source,inputs/'parent_lineage')
    else:
        shutil.copytree(source,inputs/'parent_lineage')
        verify_reference(inputs/'parent_lineage')
    initial,fixed,config=load_network(network)
    if np.any(np.asarray(fixed.bias_pA)!=0):raise ValueError('Zero-bias parent required')
    fixed,spec=joint_fit_reference(fixed)
    with np.load(readout,allow_pickle=False) as z:parent=actor_from_arrays(z)
    model=load_builtin_model()
    asset=Path(__file__).resolve().parents[1]/'r_elegans/assets/runtime_model_v1.json'
    if file_sha256(asset)!=manifest['canonical_model_asset_sha256']:
        raise ValueError('Prepared canonical neuron ordering/model asset changed')
    for w in windows:
        np.savez_compressed(inputs/(w['uid']+'-window.npz'),**{k:v for k,v in w.items() if k not in ('uid','source_head_chord_max_difference_rad')})
    settings=dict(experiment='E050',status='exploratory_joint_statistics_not_source_condition_fit',
        source=str(source),network_sha256=file_sha256(network),readout_sha256=file_sha256(readout),
        prepared_manifest_sha256=file_sha256(args.prepared_data/'manifest.json'),
        split_sha256=manifest['split_sha256'],uids=args.uids,fold='train',
        source_spatial_checks={w['uid']:w['source_head_chord_max_difference_rad'] for w in windows},
        network_config=asdict(config),trainable=spec.counts,
        fixed=['LEAK','capacitance','channel kinetics','reversal potentials','calcium and indicator kinetics',
            'initial voltages and standard flat body','NMJ and muscle parameters','mechanics','sensory gains','anatomy'],
        observation=dict(source_step_m=SOURCE_STEP_M,source_Julia_points=SOURCE_POINTS.tolist(),
            spatial='Unsigned chord bends, approximate nominal arclength; no chirality or joint inversion',
            calcium='Relative brightness = fixed basal + Hill/lag indicator; compare CV and lag-1 correlation',
            scales='0.1 CV, 0.1 rad, 0.5 correlation are optimization weights, not calibrated errors',
            sample_alignment='Midpoint exposure bins inferred from source timestamps, or point-sample sensitivity; source targets untouched; true exposure/ROI times unavailable'),
        conditions=dict(mechanics='1.7% wet-agar empirical law and/or liquid RFT; neither calibrated to mounted 2% agar',
            initial_state='Same parent voltage, equilibrated gates, flat body; fixed burn-in, no per-animal state inference',
            external_drive='None; chemical transduction disabled; intrinsic recurrent dynamics and actual-body proprioception active',
            calcium='Transferred nuclear-reporter kinetics and fixed basal brightness are assumptions',
            missingness='Source imputed samples not identifiable from finite mask'),
        optimizer='Bounded seeded greedy proposals, fresh RNG; no optimizer continuation',arguments=vars(args))
    write_json(out/'configuration.json',json.loads(json.dumps(settings,default=str)))
    rollouts={m:make_rollout(fixed,config,parent,model,medium=m,dt=args.dt,
        brain_step_ms=args.brain_step_ms,burn_in=args.burn_in,sampling=args.sampling) for m in args.media}
    started=time.monotonic()
    def evaluate(offsets,functions):
        params=apply_joint_offsets(initial,offsets,spec)
        if not frozen_joint_fields_unchanged(params,initial,spec):raise RuntimeError('Frozen parameters changed')
        rows=[];traces={}
        for medium,run in functions.items():
            for w in windows:
                trace=run(params,w['time_s'])
                metrics=paired_objective(trace['indicator'][:,w['indices']],trace['bends'],
                    w['fluorescence'],w['bends'],fluorescence_valid=w['valid'],basal_brightness=args.basal_brightness)
                rows.append(dict(medium=medium,uid=w['uid'],metrics=metrics))
                traces[medium+'-'+w['uid']]=trace
        prior=.001*float(np.mean(offsets**2))
        return dict(total=float(np.mean([r['metrics']['total'] for r in rows]))+prior,prior=prior,rows=rows),traces
    rng=np.random.default_rng(args.seed)
    best=np.zeros(spec.size)
    baseline,traces=evaluate(best,rollouts)
    best_result=baseline
    def save_best(iteration,traces):
        save_network(out/'network_best.npz',apply_joint_offsets(initial,best,spec),fixed,config)
        np.savez_compressed(out/'search_state.npz',offsets=best,iteration=iteration)
        write_json(out/'rng_state.json',rng.bit_generator.state)
        write_json(out/'best.json',dict(iteration=iteration,result=best_result))
        for name,trace in traces.items():np.savez_compressed(out/(name+'-prediction.npz'),**trace)
    save_best(0,traces)
    print(json.dumps(dict(baseline=baseline,elapsed_seconds=time.monotonic()-started)),flush=True)
    completed=0
    with (out/'history.jsonl').open('w') as history:
        for i in range(1,args.iterations+1):
            if time.monotonic()-started>=args.minutes*60:break
            proposal=joint_proposal(best,spec,rng,args.sigma,args.maximum_factor)
            try:
                result,traces=evaluate(proposal,rollouts)
                accepted=result['total']<best_result['total']
                error=None
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                result=None;accepted=False;error=str(exc)
            if accepted:
                best=proposal;best_result=result;save_best(i,traces)
            completed=i
            row=dict(iteration=i,accepted=accepted,result=result,error=error,
                     elapsed_seconds=time.monotonic()-started)
            history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
            # Preserve RNG even after rejected proposals; exact search replay needs it.
            write_json(out/'rng_state.json',rng.bit_generator.state)
            np.savez_compressed(out/'search_state.npz',offsets=best,iteration=i)
            print(json.dumps(row,allow_nan=False),flush=True)
    refined=None
    refined_baseline=None
    if args.refine:
        functions={m:make_rollout(fixed,config,parent,model,medium=m,dt=args.dt/2,
            brain_step_ms=args.brain_step_ms/2,burn_in=args.burn_in,sampling=args.sampling) for m in args.media}
        refined_baseline,_=evaluate(np.zeros(spec.size),functions)
        refined,_=evaluate(best,functions)
    summary=dict(experiment='E050',status='completed',iterations=completed,baseline=baseline,best=best_result,
        refined=refined,refined_baseline=refined_baseline,elapsed_seconds=time.monotonic()-started,parameters_changed=int(np.count_nonzero(best)),
        frozen_fields_unchanged=True,checkpoint_sha256=file_sha256(out/'network_best.npz'),
        biological_fidelity_qualified=False,heldout_evaluation_performed=False,
        stop_reason='time_budget' if completed<args.iterations else 'iterations')
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,allow_nan=False),flush=True)


if __name__=='__main__':main()
