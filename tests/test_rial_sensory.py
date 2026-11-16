from copy import deepcopy
import json,sys,time
import numpy as np
import pytest
from r_elegans.rial_sensory import protocol,features,score,parameter_groups,replay_probe,CELLS


def paired(variant='nominal',effect=.15,mode='normal'):
    p=protocol(variant);t=np.arange(0,p['duration']+.005,.01)
    ids=('RIAL','RIAR','SMDVL','SMDVR','AWAL','AWAR','AIYL');signal=np.sin(2*np.pi*1.8*t)
    traces={};result={}
    for condition in ('no_cue','cue'):
        gain=np.ones(len(t));phase=np.zeros(len(t))
        if condition=='cue':
            for a,b in p['response']:
                if mode=='phase':phase[t>=a]=.7
                elif mode!='blocked':gain[(t>=a)&(t<b)]=1-effect
            if mode=='no_recovery':gain[t>=p['events'][0][0]]=1-effect
        if mode=='weak':gain*=.5
        trace=dict(dense_time_s=t,dense_joints=.01*gain[:,None]*np.sin(2*np.pi*1.8*t[:,None]+phase[:,None])*np.ones((1,11)),
                   dense_voltage_mV=-40+3*signal[:,None]*np.ones((1,len(ids))))
        traces[condition]=trace;result[condition]=features(trace,ids,variant)
    return result,traces,ids


def test_amplitude_objective_needs_cue_specific_change_and_recovery_not_phase():
    parent=paired()[0]['no_cue']
    for variant in ('nominal','low-dose','shift','repeat','redundant','long-cue'):
        data,_,_=paired(variant);m=score(data,parent,.1)
        assert m['partial_attenuation_passed'] and not m['full_start_stop_qualified']
    for mode in ('phase','blocked','weak','no_recovery'):
        data,_,_=paired(mode=mode);assert not score(data,parent,.02)['partial_attenuation_passed']
    # Small amplitude changes guide loss before the2% gate is met.
    losses=[score(paired(effect=e)[0],parent,.02,1.3)['loss'] for e in (0.,.005,.01,.02)]
    assert all(a>b for a,b in zip(losses,losses[1:]))


def test_group_schema_freezes_motor_cells_rial_smd_edges_and_gaps():
    from test_cma_curriculum import grouped_fixture
    from r_elegans.cma_curriculum import decode_groups
    spec,ids,origin,lower,upper,_=grouped_fixture();groups=parameter_groups(spec,ids,origin,lower,upper)
    used=[i for g in groups for i in g['indices']];assert len(used)==len(set(used))
    x=decode_groups(np.ones(len(groups)),groups,origin,lower,upper);ci=np.nonzero(spec.channels);ei=np.nonzero(spec.chemical)
    nc=len(ci[0]);ne=len(ei[0])
    assert all(ids[ci[0][i]] in CELLS for i in used if i<nc)
    for n in ('DB01','SMDVL','SMDVR','AS01','ASHL'):
        np.testing.assert_array_equal(x[:nc][ci[0]==ids.index(n)],0.)
    route=(ei[1]==ids.index('RIAL'))&np.isin(ei[0],[ids.index(n) for n in ('SMDDL','SMDDR','SMDVL','SMDVR')])
    np.testing.assert_array_equal(x[nc:nc+ne][route],0.);np.testing.assert_array_equal(x[nc+ne:],0.)


def test_setup_reconstructs_selected_parent_in_original_coordinates(tmp_path,monkeypatch):
    import scripts.rial_sensory_worker as worker
    import scripts.neural_timing_setup as original
    import r_elegans.brain.conductance_network as network
    import r_elegans.rhythm_comparison as checks
    x=np.array([.1,-.2]);np.savez_compressed(tmp_path/'sensory_parent_coordinates.npz',offsets=x)
    fixed=object();config=object();decode=lambda y:(y.copy(),)
    monkeypatch.setattr(original,'setup',lambda p:dict(origin=np.zeros(2),base=(np.zeros(2),),decode=decode,fixed=fixed,config=config))
    monkeypatch.setattr(network,'load_network',lambda p:((x.copy(),),fixed,config))
    monkeypatch.setattr(checks,'assert_fixed_equal',lambda *args:None)
    d=worker.setup(tmp_path);np.testing.assert_array_equal(d['origin'],x)
    assert d['decode'] is decode
    monkeypatch.setattr(network,'load_network',lambda p:((x+1.,),fixed,config))
    with pytest.raises(ValueError,match='reconstruction'):worker.setup(tmp_path)


def test_no_cue_replay_is_final_diagnostic_on_existing_branch_only():
    ids=('RIAL','SMDVL','SMDVR','DB01');mask=np.zeros((4,4));mask[1:3,0]=1
    t=np.linspace(0,70,701);release=.3+.1*np.sin(2*np.pi*1.8*t)
    probe=replay_probe(ids,mask,t,release,40.,.002)
    before=probe(30.);after=probe(45.)
    assert not bool(before.active) and bool(after.active)
    np.testing.assert_array_equal(after.post_indices,[1,2]);np.testing.assert_array_equal(after.pre_indices,0)
    np.testing.assert_allclose(after.release_probability,np.interp(45.001,t,release),rtol=1e-5)


@pytest.mark.parametrize('objective',['body','rial-release','relay-range'])
def test_worker_training_uses_only_bath_and_preserves_models_without_raw_traces(tmp_path,monkeypatch,objective):
    import scripts.rial_sensory_worker as worker
    import scripts.compare_rhythm_worker as verifier
    import r_elegans.brain.conductance_network as network
    data,traces,ids=paired(effect=.15);calls=[]
    def factory(dt,brain,**kwargs):
        assert not any(k.startswith('diagnostic') for k in kwargs)
        cue=bool(np.asarray(kwargs['world_override'].concentration_um)[:,0].max()>0);calls.append(cue)
        return lambda *args:traces['cue' if cue else 'no_cue']
    monkeypatch.setattr(worker,'_CACHE',{})
    monkeypatch.setattr(worker,'_DATA',dict(origin=np.zeros(1),lower=-np.ones(1),upper=np.ones(1),groups=[dict(indices=[0],lower=-.4,upper=.4)],
        decode=lambda x:x,fixed=type('F',(),{'release_threshold_mV':np.full(len(ids),-40.),'release_slope_mV':np.full(len(ids),5.)})(),config=None,model=type('M',(),{'neuron_ids':ids})(),factory=factory,gains=np.ones(3),root=tmp_path))
    monkeypatch.setattr(verifier,'verify_inputs',lambda root:None)
    monkeypatch.setattr(network,'save_network',lambda path,*args:path.write_bytes(b'model'))
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda path:type('D',(),{'free':1024**4})())
    r=worker.evaluate(dict(z=[0.],output=str(tmp_path/'candidate'),until=time.monotonic()+60,target=.1,objective=objective))
    assert r['status']=='completed'
    assert r['metrics']['partial_attenuation_passed']==(objective=='body')
    if objective=='rial-release':assert r['metrics']['metric']=='RIAL_release_cycle_excursion_ratio'
    assert calls==[False,True] and (tmp_path/'candidate/network.npz').exists()
    assert not (tmp_path/'candidate/no_cue.npz').exists()


def test_eight_hour_watchdog_dispatch(tmp_path,monkeypatch):
    import scripts.total_job_budget as runner
    out=tmp_path/'out';seen=[]
    def run(cmd,seconds):
        seen.append(cmd);(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,validation_complete=True)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','rial-sensory','--output-dir',str(out),'--source-run',str(tmp_path/'source'),'--total-minutes','480'])
    assert runner.main()==0 and seen[0][3]=='scripts.run_rial_sensory'
    assert json.loads((out/'deadline.json').read_text())['experiment']=='E085'
    assert json.loads((out/'budget_receipt.json').read_text())['total_minutes']==480


def test_worker_sensory_block_and_rial_replay_are_separate_final_controls(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import scripts.rial_sensory_worker as worker
    import scripts.compare_rhythm_worker as verifier
    import r_elegans.brain.conductance_network as network
    _,traces,ids=paired(effect=.15);calls=[];mask=np.zeros((len(ids),len(ids)));mask[2:4,0]=1
    fixed=SimpleNamespace(release_threshold_mV=np.full(len(ids),-40.),release_slope_mV=np.full(len(ids),5.),chemical_mask=mask)
    def factory(dt,brain,**kwargs):
        cued=bool(np.asarray(kwargs['world_override'].concentration_um)[:,0].max()>0)
        replay=kwargs.get('diagnostic_chemical_release');calls.append((cued,replay is not None))
        if replay is not None:
            assert not bool(replay(30.).active) and bool(replay(45.).active)
            np.testing.assert_array_equal(replay(45.).post_indices,[2,3])
        def run(params,times,gains):
            np.testing.assert_array_equal(gains[1:],[1.,1.])
            return traces['cue' if cued and gains[0]>0 and replay is None else 'no_cue']
        return run
    monkeypatch.setattr(worker,'_CACHE',{})
    monkeypatch.setattr(worker,'_DATA',dict(origin=np.zeros(1),lower=-np.ones(1),upper=np.ones(1),groups=[dict(indices=[0],lower=-.4,upper=.4)],
        decode=lambda x:x,fixed=fixed,config=None,model=SimpleNamespace(neuron_ids=ids),factory=factory,gains=np.ones(3),root=tmp_path))
    monkeypatch.setattr(verifier,'verify_inputs',lambda root:None)
    monkeypatch.setattr(network,'save_network',lambda path,*args:path.write_bytes(b'model'))
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda path:SimpleNamespace(free=1024**4))
    until=time.monotonic()+60;results={}
    for mode in ('normal','sensory-blocked','rial-replay'):
        results[mode]=worker.evaluate(dict(z=[0.],output=str(tmp_path/mode),until=until,target=.1,mode=mode))
        assert results[mode]['status']=='completed'
    assert results['normal']['metrics']['partial_attenuation_passed']
    np.testing.assert_allclose(results['sensory-blocked']['metrics']['response_amplitude_ratios'],1.)
    assert results['rial-replay']['mediation']['cue_attenuation_reduced']
    assert calls==[(False,False),(True,False),(False,True),(True,True)]


@pytest.mark.parametrize('mediation_passes',[True,False])
@pytest.mark.parametrize('objective,continued',[('body',False),('rial-release',False),('relay-range',False),('relay-range',True)])
def test_curriculum_advances_and_final_tests_use_same_selected_model(tmp_path,monkeypatch,mediation_passes,objective,continued):
    import hashlib,concurrent.futures,os
    import scripts.run_rial_sensory as runner
    import scripts.compare_rhythm_worker as verifier
    import scripts.benchmark_cpu as cpu_module
    expected_parent=runner.RELAY_CONTINUATION_PARENT if continued else runner.RELAY_PARENT if objective=='relay-range' else runner.RELEASE_PARENT if objective=='rial-release' else runner.PARENT
    source=tmp_path/'source';(source/'inputs').mkdir(parents=True);folder=source/'ordinary';folder.mkdir()
    (source/'inputs/manifest.json').write_text(json.dumps(dict(sha256={},sources={})))
    for name in ('network.npz','coordinates.npz','sham.npz','no_cue.npz'):(folder/name).write_bytes(b'fixture')
    (source/'summary.json').write_text(json.dumps(dict(experiment='E087' if continued else 'fixture',selected=dict(checkpoint_sha256=expected_parent),validation=[dict(variant='nominal' if objective!='body' else 'ordinary',grid='ordinary',mode='normal',output=str(folder),checkpoint_sha256=expected_parent,trajectory_sha256=dict(sham=expected_parent,no_cue=expected_parent))])))
    out=tmp_path/'output';out.mkdir();(out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+28800,total_minutes=480)))
    monkeypatch.setattr(verifier,'verify_inputs',lambda p:None)
    monkeypatch.setattr(runner,'file_sha256',lambda p:expected_parent)
    monkeypatch.setattr(runner,'capture_run_record',lambda *args,**kwargs:None)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:type('D',(),{'free':1024**4})())
    monkeypatch.setattr(runner,'setup',lambda p:dict(spec=None,model=type('M',(),{'neuron_ids':()})(),origin=np.zeros(10),lower=-np.ones(10),upper=np.ones(10)))
    monkeypatch.setattr(runner,'parameter_groups',lambda *args:[dict(name=str(i),indices=[i],lower=-.4,upper=.4) for i in range(10)])
    import r_elegans.relay_range as relay_module
    monkeypatch.setattr(relay_module,'parameter_groups',runner.parameter_groups)
    monkeypatch.setattr(os,'sched_getaffinity',lambda n:set(range(16)),raising=False)
    monkeypatch.setattr(cpu_module,'physical_core_cpus',lambda p:(list(range(16)),True))
    model_effect={}
    def fake(job):
        path=__import__('pathlib').Path(job['output']);path.mkdir(parents=True)
        point=tuple(job['z']);effect=model_effect.setdefault(point,0. if job['reference'] else job['target']*1.31)
        data,_,_=paired(job['variant'],effect=effect,mode='blocked' if job['mode']=='sensory-blocked' else 'normal')
        if objective=='relay-range':
            from r_elegans.relay_range import score as scorer
            for name,condition in data.items():
                for category in ('active','response','late'):
                    for row in condition[category]:
                        value=.99-effect
                        if name=='cue' and category=='response' and job['mode']!='sensory-blocked':value+=effect*.001
                        row.update(AIYL_release_mean=value,AIYL_release_q01=value,AIYL_release_q99=value,AIYL_release_half_means=[value,value])
        elif objective=='rial-release':
            from r_elegans.rial_release import score as scorer
            for condition in data.values():
                for category in ('active','response','late'):
                    for row in condition[category]:row['RIAL_release_cycle_excursion']=10*row['amplitude']
        else:scorer=score
        m=scorer(data,job['parent_features'] or data['no_cue'],job['target'],job['margin'])
        payload=f'model-effect={effect:.12f}'.encode();sha=hashlib.sha256(payload).hexdigest()
        r=dict(status='completed',metrics=m,paired_features=data,checkpoint_sha256=sha,output=str(path),elapsed_seconds=1,
               variant=job['variant'],grid=job['grid'],mode=job['mode'],mediation=dict(cue_attenuation_reduced=mediation_passes) if job['mode'] in ('rial-replay','relay-replay') else None)
        (path/'network.npz').write_bytes(payload);(path/'coordinates.npz').write_bytes(b'fixture')
        (path/'request.json').write_text(json.dumps(job));(path/'summary.json').write_text(json.dumps(r));return r
    monkeypatch.setattr(runner,'evaluate',fake)
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,j):
            f=concurrent.futures.Future();f.set_result(fn(j));return f
    monkeypatch.setattr(concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(sys,'argv',['fit','--output-dir',str(out),'--source-run',str(source),'--objective',objective]+(['--continue-relay'] if continued else []))
    runner.main();s=json.loads((out/'summary.json').read_text())
    assert s['validation_complete'] and s['completed_generations']==3
    if continued:
        assert s['experiment']=='E088'
        assert json.loads((out/'configuration.json').read_text())['seed']==8800
    assert s['partial_sensory_component_qualified']==(objective=='body')
    assert s['sensory_RIAL_release_component_qualified']==(objective=='rial-release')
    assert s['sensory_relay_range_component_qualified']==(objective=='relay-range')
    assert s['final_target']==(.05 if objective=='rial-release' else .1)
    assert s['RIAL_mediated_partial_component_qualified']==(mediation_passes and objective=='body')
    assert not s['full_start_stop_qualified'] and not s['long_cue_sustained_attenuation']
    assert [r['target'] for r in s['milestones'] if r['passed']]==([.01,.02,.05] if objective=='rial-release' else [.02,.05,.1])
    assert len(s['validation'])==10 and (out/'optimizer/final/optimizer.pkl').exists()
    assert len({r['checkpoint_sha256'] for r in s['validation']})==1
