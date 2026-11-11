"""E059 matched-grid and E060 split-step read-only low-dose cue assays."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

from scripts.experiment_records import capture_run_record,file_sha256
from scripts.fit_paired_recordings import write_json


def integration_cases(mode,dt,brain_dt):
    """Return label, body step, neural step and within-branch reference."""
    if mode=='matched':
        return [('coarse',dt,brain_dt,None),('half',dt/2,brain_dt/2,'coarse'),
                ('quarter',dt/4,brain_dt/4,'half')]
    if mode=='split':
        return [('coarse',dt,brain_dt,None),('brain-half',dt,brain_dt/2,'coarse'),
                ('brain-quarter',dt,brain_dt/4,'brain-half'),
                ('body-half',dt/2,brain_dt,'coarse'),('body-quarter',dt/4,brain_dt,'body-half')]
    raise ValueError('Unknown step-assay mode')


def cue_effect_difference(coarse,fine):
    """Compare cue-minus-sham joints on coarse timestamps, all11 joints."""
    import numpy as np
    a=coarse['active'];b=fine['active']
    x=a['dense_joints']-coarse['sham']['dense_joints']
    y=b['dense_joints']-fine['sham']['dense_joints']
    ta=a['dense_time_s'];tb=b['dense_time_s']
    if ta[0]<tb[0] or ta[-1]>tb[-1]+1e-12:raise ValueError('Fine grid must cover coarse grid')
    aligned=np.column_stack([np.interp(ta,tb,y[:,j]) for j in range(11)])
    error=np.max(np.abs(x-aligned),axis=0)
    return dict(maximum_cue_effect_difference_rad=float(error.max()),per_joint_difference_rad=error.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--expected-sha256')
    p.add_argument('--mode',choices=['matched','split'],default='matched')
    args=p.parse_args()
    os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION','highest')
    import numpy as np
    import jax
    from scripts.train_sensorimotor import verified_parent
    from scripts.sensorimotor_resume import read_completed
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.brain.joint_autonomous_fit import apply_joint_offsets
    from r_elegans.brain.paired_fitting import make_rollout
    from r_elegans.body.embodied_sensory import EmbodiedSensoryConfig
    from r_elegans.sensorimotor_curriculum import ResponseProtocol,cue_world,response_metrics
    from r_elegans.model import load_builtin_model
    from r_elegans.data.paths import get_data_root
    from r_elegans.full_body_assessment import write_motion_preview
    source=args.run.resolve();cfg=json.loads((source/'configuration.json').read_text())
    initial,fixed,config,spec,_,actor,_,_,_=verified_parent(source/'parent_run')
    offsets,sensory_offsets,iteration,_,source_summary=read_completed(source,cfg,spec.size)
    sha=source_summary['checkpoint_sha256']
    if args.expected_sha256 and sha!=args.expected_sha256:raise ValueError('Expected pinned selected checkpoint')
    params,saved_fixed,saved_config=load_network(source/'network_best.npz')
    expected=apply_joint_offsets(initial,offsets,spec)
    if saved_config!=config or any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(
            jax.tree_util.tree_leaves((expected,fixed)),jax.tree_util.tree_leaves((params,saved_fixed)))):
        raise ValueError('Selected checkpoint/offset/fixed fields mismatch')
    out=get_data_root(args.output_dir,must_exist=False)
    if out==source or source in out.parents or out in source.parents:p.error('Separate output required')
    if any(path.is_symlink() for path in source.rglob('*')):raise ValueError('Parent archive contains symlinks')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,Path(__file__).resolve().parents[1],args)
    shutil.copytree(source,out/'parent_run')
    sensory_sha=file_sha256(source/'sensory_parameters.json')
    if file_sha256(out/'parent_run/network_best.npz')!=sha or file_sha256(out/'parent_run/sensory_parameters.json')!=sensory_sha:
        raise ValueError('Copied selected model mismatch')
    protocol=ResponseProtocol(**cfg['protocol']);sc=EmbodiedSensoryConfig(**cfg['base_sensory_config'])
    model=load_builtin_model();gains=np.exp(sensory_offsets)
    times=np.linspace(0,protocol.duration_s,81)
    dt=cfg['arguments']['dt'];brain_dt=cfg['arguments']['brain_step_ms']
    cases=integration_cases(args.mode,dt,brain_dt)
    experiment='E059' if args.mode=='matched' else 'E060'
    write_json(out/'configuration.json',dict(experiment=experiment,mode=args.mode,parent_checkpoint_sha256=sha,
        sensory_parameters_sha256=sensory_sha,source_run=str(source),source_stage_pass=source_summary['passed'],
        protocol=asdict(protocol),dose_um=.8,gain_multipliers=gains.tolist(),base_sensory_config=asdict(sc),
        integration_cases=[dict(case=n,dt_s=d,brain_step_ms=b,reference=r) for n,d,b,r in cases],
        body_steps_s=[d for n,d,b,r in cases],brain_steps_ms=[b for n,d,b,r in cases],
        fixed='All model and transduction parameters; no optimization or RNG draws',
        coupling_scope='Body step also changes sensory updates and co-simulation input hold; neural step varies internal frozen-conductance updates',
        scope='Engineering sensitivity and margin diagnostics, not certified numerical error or biological qualification'))
    rows=[];comparisons=[];recordings={}
    for label,body_step,neural_step,reference in cases:
        destination=out/label;destination.mkdir();traces={}
        print(f'Starting low-dose {label}: dt={body_step}, brain_step_ms={neural_step}',flush=True)
        for name,sham in [('active',False),('sham',True)]:
            run=make_rollout(fixed,config,actor,model,medium='wet-agar',dt=body_step,
                brain_step_ms=neural_step,burn_in=0.,sampling='point',record_dense=True,
                sensory_config=sc,chemical_world=cue_world(protocol,dose=.8,sham=sham))
            traces[name]=run(params,times,gains)
            np.savez_compressed(destination/(name+'-trajectory.npz'),**traces[name])
        metrics=response_metrics(traces['active'],traces['sham'],model.neuron_ids,protocol)
        write_json(destination/'metrics.json',metrics)
        trace=traces['active']
        write_motion_preview(destination/'motion.html',trace['dense_joints'],trace['dense_position'],trace['dense_heading'],
            dt=body_step,body_length=.001,seconds=protocol.duration_s,title=experiment+' low-dose '+label+'; fixed checkpoint')
        row=dict(case=label,dt_s=body_step,brain_step_ms=neural_step,metrics=metrics,
            target_margin_rad=metrics['cue_specific_peak_rad']-protocol.bend_target_rad)
        rows.append(row)
        if reference is not None:comparisons.append(dict(pair=reference+'->'+label,**cue_effect_difference(recordings[reference],traces)))
        recordings[label]=traces
        print(json.dumps(dict(case=label,passed=metrics['passed'],bend_rad=metrics['cue_specific_peak_rad'],
            recovery=metrics['recovery_ratio'],awa_mV=metrics['awa_depolarization_mV'],target_margin_rad=row['target_margin_rad'])),flush=True)
    margin=min(r['target_margin_rad'] for r in rows[1:]);last_error=comparisons[-1]['maximum_cue_effect_difference_rad']
    result=dict(experiment=experiment,mode=args.mode,status='completed',source_run=str(source),checkpoint_sha256=sha,
        sensory_parameters_sha256=sensory_sha,source_stage_pass=source_summary['passed'],global_iteration=iteration,
        cases=rows,comparisons=comparisons,all_grids_response_pass=all(r['metrics']['passed'] for r in rows),
        all_successive_differences_within_existing_tolerance=all(c['maximum_cue_effect_difference_rad']<=.25*protocol.bend_target_rad for c in comparisons),
        margin_scope='Conservative descriptive confidence check, not a new retrospective stage gate or rigorous error bound',
        parameters_changed=False,promotion_automatic=False)
    if args.mode=='matched':
        result.update(minimum_fine_grid_threshold_margin_rad=margin,
            fine_margin_exceeds_last_successive_difference=bool(margin>last_error))
    else:
        bycase={r['case']:r['metrics'] for r in rows}
        result['branch_peak_ratios_to_coarse']={branch:[bycase[branch+'-'+step]['cue_specific_peak_rad']/max(bycase['coarse']['cue_specific_peak_rad'],1e-30)
            for step in ('half','quarter')] for branch in ('brain','body')}
        result['interpretation']='Brain-only step versus body/sensory/coupling step; not a causal attribution to mechanics alone or certified convergence'
    if file_sha256(source/'network_best.npz')!=sha or file_sha256(source/'sensory_parameters.json')!=sensory_sha:
        raise ValueError('Source model changed')
    write_json(out/'summary.json',result)
    print('All-grid low-dose response pass:',result['all_grids_response_pass'],flush=True)
    print('Successive cue-effect differences:',json.dumps(comparisons),flush=True)
    if args.mode=='matched':
        print('Fine margin exceeds last grid difference:',result['fine_margin_exceeds_last_successive_difference'],flush=True)
    else:
        print('Brain/body peak ratios to coarse:',json.dumps(result['branch_peak_ratios_to_coarse']),flush=True)
    print('Report:',out/'summary.json',flush=True)


if __name__=='__main__':main()
