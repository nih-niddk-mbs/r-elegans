"""E075: audit saved E074 signal flow without simulation or parameter changes."""
import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np
from scipy.special import expit

from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.control_signal_audit import paired_summary,source_currents,replay_chemical
from r_elegans.behavior_control import CONTROL_GROUPS,ControlProtocol,bath_schedule

EXPECTED='2a2b30b4324163aaba5e6c70d4d549cf0c689732d03ebb65f03aec297ee0924f'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    for source in (repo,run):
        if out==source or source in out.parents or out in source.parents:p.error('Separate external output required')
    # Check the entire frozen input capsule and every sampled trajectory hash.
    from scripts.compare_rhythm_worker import verify_inputs
    manifest=verify_inputs(run/'control_inputs')
    gate=json.loads((run/'control/control_gate.json').read_text())
    summary=json.loads((run/'summary.json').read_text())
    selected=Path(next(r['selected'] for r in summary['phases'] if r['stage']=='control'))
    if not selected.is_relative_to(run/'control/fit'):raise ValueError('Selected fit outside pinned run')
    model=selected/'network_best.npz'
    if gate['checkpoint_sha256']!=EXPECTED or file_sha256(model)!=EXPECTED:raise ValueError('Pinned E074 checkpoint mismatch')
    inputs=[model,selected/'best_state.npz',selected/'parameter_origin.npz',run/'summary.json',run/'control/control_gate.json',run/'control_inputs/manifest.json']
    traces={}
    for case in ('nominal','no-cue','chemical-blocked','stop-blocked','start-blocked'):
        folder=run/'control/validate'/case
        report=json.loads((folder/'summary.json').read_text());trace=folder/'trajectory.npz'
        if report['checkpoint_sha256']!=EXPECTED or report['trajectory_sha256']!=file_sha256(trace):raise ValueError('Saved case mismatch: '+case)
        with np.load(trace,allow_pickle=False) as z:
            traces[case]={k:z[k] for k in ('dense_time_s','dense_voltage_mV','dense_calcium_uM','dense_muscles','dense_command','dense_filtered_drive','dense_joints')}
        inputs.extend([folder/'summary.json',trace])
    t=traces['nominal']['dense_time_s']
    for case,tr in traces.items():
        if not np.array_equal(t,tr['dense_time_s']):raise ValueError('Unmatched grid: '+case)
    hashes={str(f):file_sha256(f) for f in inputs}
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,a)
    for name,path in [('network.npz',model),('coordinates.npz',selected/'best_state.npz'),('origin.npz',selected/'parameter_origin.npz')]:
        shutil.copy2(path,out/name)
        if file_sha256(path)!=file_sha256(out/name):raise ValueError('Audit snapshot mismatch')
    import jax
    jax.config.update('jax_enable_x64',True)
    from r_elegans.brain.conductance_network import load_network,network_synapses
    from r_elegans.model import load_builtin_model
    params,fixed,config=load_network(model);ids=load_builtin_model().neuron_ids
    chemical,gap=map(np.asarray,network_synapses(params,fixed,config))
    groups={'chemical_sensory':('AWAL','AWAR','AWCL','AWCR','ASEL','ASER'),**CONTROL_GROUPS,
        'body_motor':tuple(n for n in ids if re.fullmatch(r'(DA|DB|DD|VA|VB|VD|AS)\d+',n)),
        'head_motor':tuple(n for n in ids if n.startswith(('RMD','SMD','SMB')))}
    named_neural={};rankings={};other={};currents={};topology={}
    for case in ('nominal','chemical-blocked','stop-blocked','start-blocked'):
        cue=traces[case];sham=traces['no-cue'];groups_result={}
        dv=np.max(abs(cue['dense_voltage_mV'][(t>=40)&(t<68)]-sham['dense_voltage_mV'][(t>=40)&(t<68)]),axis=0)
        rankings[case]=[dict(neuron=ids[i],maximum_voltage_difference_mV=float(dv[i])) for i in np.argsort(-dv)[:20]]
        for group,names in groups.items():
            indices=[ids.index(n) for n in names]
            groups_result[group]={}
            for field in ('dense_voltage_mV','dense_calcium_uM'):
                groups_result[group][field]=paired_summary(t,cue[field][:,indices],sham[field][:,indices],names)
            for condition,tr in [('cue',cue),('sham',sham)]:
                release=expit((tr['dense_voltage_mV'][:,indices]-np.asarray(fixed.release_threshold_mV)[indices])/np.asarray(fixed.release_slope_mV)[indices])
                groups_result[group][condition+'_release_range']=[dict(neuron=n,minimum=float(release[:,j].min()),maximum=float(release[:,j].max())) for j,n in enumerate(names)]
        named_neural[case]=groups_result
        other[case]={field:paired_summary(t,cue[field],sham[field],[str(j) for j in range(cue[field].shape[1])]) for field in ('dense_muscles','dense_command','dense_filtered_drive','dense_joints')}
    # Subsample recorded voltages to limit matrix work. No neural update is run.
    stride=10;clock=t[::stride];v=traces['nominal']['dense_voltage_mV'][::stride];v0=traces['no-cue']['dense_voltage_mV'][::stride]
    links=[('chemical_sensory',g) for g in CONTROL_GROUPS]+[(g,m) for g in CONTROL_GROUPS for m in ('body_motor','head_motor')]
    for src,dst in links:
        key=src+'->'+dst;si=[ids.index(n) for n in groups[src]];di=[ids.index(n) for n in groups[dst]]
        i,g=source_currents(v,chemical,gap,fixed.synapse_reversal_mV,fixed.release_threshold_mV,fixed.release_slope_mV,si,di)
        i0,g0=source_currents(v0,chemical,gap,fixed.synapse_reversal_mV,fixed.release_threshold_mV,fixed.release_slope_mV,si,di)
        currents[key]={'chemical_inward_pA':paired_summary(clock,i,i0,groups[dst]),'electrical_inward_pA':paired_summary(clock,g,g0,groups[dst])}
        topology[key]=[dict(source=ids[j],target=ids[k],chemical_nS=float(chemical[k,j]),gap_nS=float(gap[k,j]),source_reversal_mV=float(fixed.synapse_reversal_mV[j])) for k in di for j in si if chemical[k,j]>0 or gap[k,j]>0]
    # This chemical-only replay is an intended-input diagnostic, not a measured
    # conductance trace. Historical sensory memory is taken from its source capsule.
    from scripts.neural_timing_setup import setup
    d=setup(run/'control_inputs')
    from r_elegans.rhythm_comparison import assert_fixed_equal
    assert_fixed_equal(d['fixed'],fixed,d['config'],config)
    source_cfg=json.loads((run/'control_inputs/new_configuration.json').read_text())['base_sensory_config']
    from r_elegans.body.embodied_sensory import EmbodiedSensoryConfig
    sc=EmbodiedSensoryConfig(**source_cfg);memory=d['state'].sensory
    world=bath_schedule(ControlProtocol());shamworld=bath_schedule(ControlProtocol(sham=True))
    replay_args=(memory.receptor,memory.adapted,np.asarray(sc.chemical_reference_um),sc.receptor_time_constant_s,sc.adaptation_time_constant_s,sc.chemical_gain_nS*d['gains'][0])
    replay=replay_chemical(t,np.asarray(world.times_s),np.asarray(world.concentration_um),*replay_args)
    zero=replay_chemical(t,np.asarray(shamworld.times_s),np.asarray(shamworld.concentration_um),*replay_args)
    replay_report=paired_summary(t,replay,zero,groups['chemical_sensory'])
    np.savez_compressed(out/'chemical_replay.npz',time_s=t,cue_excitatory_nS=replay,sham_excitatory_nS=zero)
    with np.load(selected/'parameter_origin.npz') as z:origin=z['origin'];allowed=z['trainable'];lower=z['lower'];upper=z['upper']
    with np.load(selected/'best_state.npz') as z:x=z['offsets']
    if x.shape!=origin.shape or np.any(x<lower-1e-12) or np.any(x>upper+1e-12) or not np.array_equal(x[~allowed],origin[~allowed]):raise ValueError('Selected coordinate bounds/frozen mask failed')
    rebuilt=d['decode'](x)
    if any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(rebuilt,params)):raise ValueError('Coordinates do not reconstruct selected model')
    histories=[]
    for worker in sorted((run/'control/fit').glob('seed-*')):
        rows=0;accepted=0;errors=0;first=None;last=None
        path=worker/'history.jsonl'
        if path.exists():
            inputs.append(path);hashes[str(path)]=file_sha256(path)
            with path.open() as stream:
                for line in stream:
                    row=json.loads(line);rows+=1;accepted+=bool(row['accepted']);errors+=bool(row['error'])
                    value=dict(iteration=row['iteration'],epoch=row['epoch'],loss=row['best']['loss'],rest_ranges=[r['maximum_raw_joint_range_rad'] for r in row['best']['metrics']['off']])
                    if first is None:first=value
                    last=value
        histories.append(dict(worker=str(worker),proposals=rows,accepted=accepted,errors=errors,first_reported_best=first,last_reported_best=last))
    if any(file_sha256(Path(f))!=sha for f,sha in hashes.items()):raise ValueError('Source artifact changed during audit')
    report=dict(experiment='E075',status='completed_read_only_analysis',checkpoint_sha256=EXPECTED,
        no_forward_simulation=True,no_training=True,source_run=str(run),input_sha256=hashes,
        scope='Descriptive paired responses and sampled inferred synaptic currents; not causal mediation or biological validation',
        caveats=['Offline chemical replay is not recorded membrane injection and can differ at floating-point event boundaries',
                 'Synaptic currents inferred at saved voltage samples; no ionic gates reconstructed or substep currents measured',
                 'Cue-sham divergence may include phase shifts; waveform differences are not proof of control',
                 'Thresholds/reversals are fixed exploratory priors, not verified neuron-specific physiology',
                 'Source histories first entry is after first proposal, not an initial cued baseline'],
        pre_cue_voltage_max_difference_mV=float(np.max(abs(traces['nominal']['dense_voltage_mV'][t<40]-traces['no-cue']['dense_voltage_mV'][t<40]))),
        neural=named_neural,top_voltage_responses=rankings,downstream=other,
        sampled_source_currents=currents,topology=topology,chemical_replay=replay_report,
        parameters=dict(total=len(x),trainable=int(allowed.sum()),changed=int(np.count_nonzero(x-origin)),
            frozen_unchanged=True,factor_percentiles=np.percentile(np.exp(x[allowed]-origin[allowed]),[0,25,50,75,100]).tolist(),
            near_bound=int(np.count_nonzero((x[allowed]-lower[allowed]<.01*(upper-lower)[allowed])|(upper[allowed]-x[allowed]<.01*(upper-lower)[allowed])))),histories=histories)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Read-only audit completed:',out/'summary.json',flush=True)
    print('Checkpoint:',EXPECTED,'; pre-cue max voltage difference:',report['pre_cue_voltage_max_difference_mV'],flush=True)
    for phase in ('stop_cue','after_stop','rest','start_cue'):
        print('\nPhase:',phase,flush=True)
        for name,group in named_neural['nominal'].items():
            values=group['dense_voltage_mV'][phase];strongest=max(values,key=lambda r:r['maximum_absolute_difference'])
            print(name,'max cue-sham mV:',strongest['maximum_absolute_difference'],'at',strongest['name'],flush=True)
        print('replayed chemical peak nS:',{r['name']:r['maximum_absolute_difference'] for r in replay_report[phase]},flush=True)
    print('Parameters:',report['parameters'],flush=True)
    for path,data in currents.items():
        print('Sampled drive',path,{kind:max(r['maximum_absolute_difference'] for r in table['stop_cue']) for kind,table in data.items()},flush=True)
    print('RIS fixed chemical reversal mV:',float(fixed.synapse_reversal_mV[ids.index('RIS')]),flush=True)
    for row in histories:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
