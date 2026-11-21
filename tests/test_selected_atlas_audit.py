import ast,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.selected_atlas_audit_checks import compare_responses,parameter_audit,selection_audit

BUNDLE=Path('r_elegans/assets/mac_rhythm_v1_polarized/network.npz')


def test_comparison_does_not_use_correlation_alone_and_rejects_nonfinite():
    a=np.array([[1.,2.,3.],[4.,5.,6.]])
    assert not compare_responses(a,2*a)['passed']  # correlation1, wrong amplitude
    assert compare_responses(a,a*(1+1e-5))['passed']
    assert not compare_responses(a,np.full_like(a,np.nan))['passed']
    assert not compare_responses(a,a[:1])['passed']


def test_selected_frozen_fields_and_extra_bounds_are_checked_on_selected_model(tmp_path):
    with np.load(BUNDLE,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    selected=tmp_path/'selected.npz';np.savez(selected,**data)
    audit=parameter_audit(selected,BUNDLE,BUNDLE,'E104',dict(parameter_set='all'))
    assert audit['passed']
    data['cells_capacitance_pF']=np.full(302,19.)
    np.savez(selected,**data)
    assert parameter_audit(selected,BUNDLE,BUNDLE,'E104',dict(parameter_set='all'))['passed']
    assert not parameter_audit(selected,BUNDLE,BUNDLE,'E104',dict(parameter_set='network'))['passed']
    data['cells_capacitance_pF']=np.full(302,21.)
    np.savez(selected,**data)
    assert not parameter_audit(selected,BUNDLE,BUNDLE,'E104',dict(parameter_set='all'))['passed']
    data['fixed_synapse_reversal_mV']=data['fixed_synapse_reversal_mV']+1
    np.savez(selected,**data)
    assert any('Frozen field' in e for e in parameter_audit(selected,BUNDLE,BUNDLE,'E104',dict(parameter_set='all'))['errors'])


def test_frozen_peptide_run_must_keep_its_actual_parent_not_bundle_anchor(tmp_path):
    with np.load(BUNDLE,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    data['params_log_channel_scale']=data['params_log_channel_scale'].astype(float).copy();data['params_log_channel_scale'][0,0]+=.1
    parent=tmp_path/'parent.npz';np.savez(parent,**data)
    assert parameter_audit(parent,parent,BUNDLE,'E105',dict(train_network='no'))['passed']
    assert not parameter_audit(BUNDLE,parent,BUNDLE,'E105',dict(train_network='no'))['passed']


def test_selection_checks_logged_validation_not_test():
    summary=dict(evaluations=[dict(step=0,val=dict(spearman_mean_z=.1),test=dict(spearman_mean_z=.8)),dict(step=25,val=dict(spearman_mean_z=.2),test=dict(spearman_mean_z=.1))],best_val=dict(step=25,spearman=.2))
    assert selection_audit(summary)['passed']
    summary['best_val']=dict(step=0,spearman=.1)
    assert not selection_audit(summary)['passed']


def original_functions(path,context):
    # Use the existing trusted trainer's actual nested forward functions as the
    # reference, rather than a second handwritten copy in this test.
    source=ast.parse(Path(path).read_text());main=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    definitions=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in ('make_settled','make_responses')]
    module=ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[]));exec(compile(module,str(path),'exec'),context)
    return context['make_settled'],context['make_responses']


@pytest.mark.parametrize('peptides',[False,True])
def test_audit_forward_matches_actual_trainer_forward_without_parameter_updates(peptides):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import load_network,advance_network,initial_network_state,MembraneConductanceInput
    from r_elegans.brain.extrasynaptic import peptide_level_step,peptide_membrane_input
    from r_elegans.brain.selected_atlas_replay import atlas_forward
    p,f,c=load_network(BUNDLE);p,f=jax.tree.map(lambda x:x.astype(jnp.float64) if jnp.issubdtype(x.dtype,jnp.floating) else x,(p,f))
    dt=.002;frames=10;warmup=.02;steps=2;g=jnp.zeros((2,302)).at[0,0].set(.01).at[1,1].set(.01)
    context=dict(jax=jax,jnp=jnp,np=np,DT_S=dt,FRAMES=frames,stim_frames=2,lo=2,hi=8,REVERSAL_MV=0.,config=c,fixed=f,float_type=jnp.float64,
        advance_network=advance_network,initial_network_state=initial_network_state,MembraneConductanceInput=MembraneConductanceInput,peptide_level_step=peptide_level_step,peptide_membrane_input=peptide_membrane_input,unit=c.synaptic_unit_nS)
    make_settled,make_responses=original_functions('scripts/train_atlas_peptides.py' if peptides else 'scripts/train_all_parameters.py',context)
    # Use an explicit short reference settle and the actual trainer response
    # function; compare with the auditor's independently settled forward replay.
    # No optimizer step is invoked.
    if peptides:
        exc=jnp.eye(302)*1e-5;inh=jnp.eye(302)*2e-5;tau=jnp.log(5.);level=jnp.zeros(302)
        def step(carry,_):
            state,level=carry;membrane=peptide_membrane_input(level,exc,inh,c.synaptic_unit_nS)
            state=advance_network(state,p,f,dt,steps,c,membrane_input=membrane)
            return (state,peptide_level_step(level,state.voltage_mV,f,dt,tau)),None
        state,level=jax.jit(lambda:jax.lax.scan(step,(initial_network_state(f,c),level),None,length=10)[0])()
        expected=np.asarray(jax.jit(make_responses(steps))(p,exc,inh,tau,state,level,g))
        got=atlas_forward(p,f,c,g,steps,peptide=(exc,inh,tau),dt=dt,settle_seconds=warmup,frames=frames,stim_seconds=.004,window=(.004,.016))
    else:
        def step(state,_):return advance_network(state,p,f,dt,steps,c),None
        state=jax.jit(lambda:jax.lax.scan(step,initial_network_state(f,c),None,length=10)[0])()
        expected=np.asarray(jax.jit(make_responses(steps))(p,f,state,g))
        got=atlas_forward(p,f,c,g,steps,dt=dt,settle_seconds=warmup,frames=frames,stim_seconds=.004,window=(.004,.016))
    np.testing.assert_allclose(got,expected,rtol=1e-10,atol=1e-15)


def test_dense_peptide_rhythm_keeps_original_update_clock(monkeypatch):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import load_network,advance_network,initial_network_state
    from r_elegans.brain.extrasynaptic import peptide_membrane_input,peptide_level_step
    from r_elegans.brain.selected_atlas_replay import rhythm_forward
    p,f,c=load_network(BUNDLE);p,f=jax.tree.map(lambda x:x.astype(jnp.float64) if jnp.issubdtype(x.dtype,jnp.floating) else x,(p,f))
    exc=jnp.eye(302)*1e-5;inh=exc;tau=jnp.log(5.);state=initial_network_state(f,c);level=jnp.zeros(302);expected=[]
    for _ in range(2):
        membrane=peptide_membrane_input(level,exc,inh,c.synaptic_unit_nS)
        for _ in range(5):
            state=advance_network(state,p,f,.002,2,c,membrane_input=membrane);expected.append(np.asarray(state.voltage_mV))
        level=peptide_level_step(level,state.voltage_mV,f,.01,tau)
    got=rhythm_forward(p,f,c,peptide=(exc,inh,tau),seconds=.02,sample_dt=.002,peptide_dt=.01)
    np.testing.assert_allclose(got,expected,rtol=1e-10,atol=1e-12)


def test_source_preparation_verifies_selected_hash_and_preserves_original_files(tmp_path):
    import shutil
    from scripts.run_selected_atlas_audit import prepare_case
    from scripts.experiment_records import file_sha256
    source=tmp_path/'source';source.mkdir();shutil.copyfile(BUNDLE,source/'network_best_val.npz');shutil.copyfile(BUNDLE,source/'parent_network.npz')
    sha=file_sha256(BUNDLE);summary=dict(experiment='E104',status='completed',best_val_sha256=sha,parent_sha256=sha,anchor_sha256=sha,steps=1,parent_network=str(BUNDLE.resolve()))
    (source/'summary.json').write_text(json.dumps(summary));np.savez(source/'responses.npz',targets=['ADAL'],best_val=np.zeros((1,302)))
    (source/'history.jsonl').write_text('{"step":0,"loss":1.0}\n')
    record=source/'run_records/original';record.mkdir(parents=True)
    (record/'command.json').write_text(json.dumps(dict(parsed_arguments=dict(output_dir=str(source),experiment_id='E104',parameter_set='network'))))
    before={p.name:file_sha256(p) for p in source.iterdir() if p.is_file()}
    m=prepare_case(source,tmp_path/'inputs','E104')
    assert m['arguments']['parameter_set']=='network' and m['provenance']['history']['steps_match']
    assert not m['provenance']['original_source_archive_verified']
    assert before=={p.name:file_sha256(p) for p in source.iterdir() if p.is_file()}
    (source/'network_best_val.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):prepare_case(source,tmp_path/'bad','E104')


@pytest.mark.parametrize('reference_fails',[False,True])
def test_audit_coordinator_keeps_all_eight_cases_and_negative_outcomes(tmp_path,monkeypatch,reference_fails):
    import sys
    import scripts.run_selected_atlas_audit as runner
    from scripts.selected_atlas_audit_checks import write
    out=tmp_path/'audit';out.mkdir();jobs=[]
    def prepare(source,inputs,experiment):
        inputs.mkdir(parents=True);write(inputs/'summary.json',dict(parent_sha256='a'*64));(inputs/'parent_network.npz').write_bytes(b'parent')
        return dict(arguments=dict(seed=0,validation_every_nth=4),evidence=[])
    def stats(root,seed,every,folder):
        result=dict(metadata={},targets=['ADAL']);write(folder/'statistics.json',result);write(folder/'data_evidence.json',[]);return folder/'statistics.json',result
    class Process:
        returncode=0
        def __init__(self,command,env,stdout,stderr):
            self.returncode=2 if reference_fails and '--reference-network' in command else 0
            jobs.append(command);assert env['JAX_PLATFORMS']=='cuda' and env['CUDA_VISIBLE_DEVICES'] in ('0','1')
            assert 'scripts.audit_selected_atlas_worker' in command
            folder=Path(command[command.index('--output-dir')+1]);folder.mkdir(parents=True)
            write(folder/'summary.json',dict(audit_complete=True,integrity_and_atlas_numerics_passed=False,problems=['synthetic numerical failure']))
        def poll(self):return self.returncode
    monkeypatch.setattr(runner,'prepare_case',prepare);monkeypatch.setattr(runner,'prepare_statistics',stats)
    monkeypatch.setattr(runner,'capture_run_record',lambda *a:None);monkeypatch.setattr(runner.subprocess,'Popen',Process);monkeypatch.setattr(runner.time,'sleep',lambda n:None)
    monkeypatch.setattr(sys,'argv',['audit','--e104-sweep',str(tmp_path/'E104'),'--e105-sweep',str(tmp_path/'E105'),'--data-root',str(tmp_path/'data'),'--output-dir',str(out)])
    assert runner.main()==(2 if reference_fails else 0)
    summary=json.loads((out/'summary.json').read_text())
    assert summary['audit_complete']==(not reference_fails) and len(summary['finished'])==8
    assert not summary['all_selected_integrity_and_atlas_numerics_passed'] and not summary['model_promoted']
    assert len([j for j in jobs if '--case' in j])==8


@pytest.mark.parametrize('timed_out',[False,True])
def test_audit_watchdog_does_not_promote_or_hide_timeout(tmp_path,monkeypatch,timed_out):
    import sys
    import scripts.selected_atlas_audit_job as runner
    out=tmp_path/'audit'
    def run(command,seconds):
        assert 'scripts.run_selected_atlas_audit' in command and seconds<=120*60
        (out/'summary.json').write_text(json.dumps(dict(audit_complete=True)))
        return (124 if timed_out else 0),timed_out
    monkeypatch.setattr(runner,'capture_run_record',lambda *a:None);monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['job','--e104-sweep',str(tmp_path/'E104'),'--e105-sweep',str(tmp_path/'E105'),'--data-root',str(tmp_path/'data'),'--output-dir',str(out)])
    assert runner.main()==(124 if timed_out else 0)
    receipt=json.loads((out/'budget_receipt.json').read_text())
    assert receipt['audit_complete']==(not timed_out) and not receipt['model_promoted']


@pytest.mark.parametrize('peptides',[False,True])
def test_worker_replays_actual_selected_weights_not_the_final_checkpoint(tmp_path,monkeypatch,peptides):
    import shutil
    import scripts.audit_selected_atlas_worker as worker
    import r_elegans.brain.selected_atlas_replay as replay
    from scripts.selected_atlas_audit_checks import write
    from scripts.experiment_records import file_sha256
    from r_elegans.atlas_evaluation import pair_metrics,model_pairs
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    inputs=tmp_path/'inputs';inputs.mkdir();sha=file_sha256(BUNDLE)
    for name in ('network_best_val.npz','parent_network.npz','anchor_network.npz'):shutil.copyfile(BUNDLE,inputs/name)
    # A deliberately invalid final checkpoint must never be loaded for this audit.
    (inputs/'network_final.npz').write_bytes(b'wrong final checkpoint')
    with np.load(BUNDLE,allow_pickle=False) as z:ids=z['neuron_ids'].tolist();expected=z['params_log_channel_scale']
    targets=[ids[0]];response=np.arange(302,dtype=float)[None,:]*1e-4+.001
    stats=dict(pairs=[(ids[0],n) for n in ids[1:13]],mean_z=np.arange(12.),n=np.ones(12),stouffer=np.linspace(-4,4,12))
    values=pair_metrics(model_pairs(response,targets,ids,stats),stats,permutations=0)
    values={k:v for k,v in values.items() if not k.startswith('permutation')}
    source=dict(experiment='E104',parameter_set='network',best_val_sha256=sha,parent_sha256=sha,best_val=dict(step=0,spearman=values['spearman_mean_z']),
        evaluations=[dict(step=0,val=values)],best_val_checkpoint={split:values for split in ('train','val','test')},data=dict(animals=1))
    if peptides:
        asset_sha=file_sha256('r_elegans/assets/extrasynaptic_v1.npz')
        np.savez(inputs/'network_best_val_peptides.npz',raw_exc=np.full(269,-10.),raw_inh=np.full(269,-10.),log_tau_s=np.log(5.),asset_sha256=asset_sha)
        np.savez(inputs/'network_final_peptides.npz',raw_exc=np.full(269,2.),raw_inh=np.full(269,2.),log_tau_s=np.log(30.))
        source.update(experiment='E105',peptide_asset_sha256=asset_sha,best_val_peptides_sha256=file_sha256(inputs/'network_best_val_peptides.npz'),best_val_peptides_off=source['best_val_checkpoint'])
    write(inputs/'summary.json',source);np.savez(inputs/'responses.npz',targets=targets,best_val=response,best_val_peptides_off=response,stimulus_nS=.01)
    cache=tmp_path/'statistics.json';write(cache,dict(stats={split:stats for split in ('train','val','test')}))
    reference=tmp_path/'reference.json';t=np.arange(2000)*.002;voltage=np.tile((-40+10*np.sin(2*np.pi*5*t))[:,None],(1,302))
    write(reference,dict(checkpoint_sha256=sha,cells={n:rhythm_metrics(voltage[:,i],.002) for i,n in enumerate(ids) if n.startswith('RMD')}))
    manifest=dict(arguments=dict(parameter_set='network',train_network='no',resettle_every=10,seed=0,substeps=40,stimulus_nS=.01),
        statistics=dict(metadata=dict(animals=1),targets=targets),evidence=[dict(path=str(p),sha256=file_sha256(p)) for p in inputs.iterdir() if p.name not in ('network_final.npz','network_final_peptides.npz')],
        provenance=dict(original_source_archive_verified=True,replay_source_checks=[dict(matched=True)],history=dict(available=True,steps_match=True,finite=True)),historical_limits=[])
    if peptides:manifest['peptide_asset_sha256']=asset_sha
    write(inputs/'manifest.json',manifest);calls=[]
    def forward(p,f,c,g,substeps,**kwargs):
        np.testing.assert_array_equal(p.log_channel_scale,expected);calls.append(substeps);return response[:len(g)]
    monkeypatch.setattr(replay,'atlas_forward',forward);monkeypatch.setattr(replay,'rhythm_forward',lambda *a,**k:voltage)
    monkeypatch.setattr(worker,'capture_run_record',lambda *a:None)
    case=dict(label='selected',inputs=str(inputs),source=str(tmp_path/'source'),manifest=str(inputs/'manifest.json'),manifest_sha256=file_sha256(inputs/'manifest.json'),statistics=str(cache),reference=str(reference))
    out=tmp_path/'result';worker.run_case(case,out)
    result=json.loads((out/'summary.json').read_text())
    assert result['audit_complete'] and result['integrity_and_atlas_numerics_passed']
    assert result['checkpoint_sha256']==sha and calls==([40,80,40,80] if peptides else [40,80])
    if peptides:assert result['checks']['peptides']['tau_s']==pytest.approx(5.)
    assert result['checks']['rhythm']['reference_checkpoint_sha256']==sha
    assert (inputs/'network_final.npz').read_bytes()==b'wrong final checkpoint'


def test_original_performance_gates_keep_ablation_direction_and_numerical_limit():
    from scripts.selected_atlas_audit_checks import declared_performance
    r=dict(source_experiment='E105',selected_metrics=dict(test=dict(spearman_mean_z=.15,auc_excited=.6)),
        selected_peptide_off_metrics=dict(test=dict(spearman_mean_z=.18,auc_excited=.6)),integrity_and_atlas_numerics_passed=True,validation_leakage=False)
    assert not declared_performance(r)['passed']
    r['selected_peptide_off_metrics']['test']['spearman_mean_z']=.12
    r['integrity_and_atlas_numerics_passed']=False
    assert declared_performance(r)['passed'] and not declared_performance(r)['numerical_interpretation_allowed']
