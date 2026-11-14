from copy import deepcopy
import pickle
import numpy as np
import pytest
from r_elegans.cma_curriculum import CONDITIONS,condition_protocol,features,score,parameter_groups,decode_groups


def paired_features(target=.05,mode='controlled',variant='nominal'):
    result={}
    for condition in CONDITIONS:
        p=condition_protocol(condition,variant);on,off=p.windows()
        f0=1.8;slow=f0*(1-target)
        def row(window,f):return dict(window=list(window),frequency_hz=[f]*3,regional_cycles_rad=[.02]*3,maximum_peak_gap_s=[1/f]*3,maximum_raw_range_rad=.02)
        rows_on=[]
        for k,w in enumerate(on):
            f=slow if condition=='stop_only' and k>0 else f0
            if mode=='no_restart' and condition=='stop_start' and k>0:f=slow
            if mode=='spontaneous' and condition=='stop_only':f=f0
            if mode=='always_moving':f=f0
            rows_on.append(row(w,f))
        rows_off=[row(w,slow if condition in ('stop_only','stop_start') and mode!='always_moving' else f0) for w in off]
        result[condition]=dict(on=rows_on,off=rows_off,safe=True)
    return result


def test_paired_cadence_requires_slowing_and_start_specific_recovery():
    paired=paired_features();parent=paired['no_cue']
    assert score(paired,parent,.05)['partial_cadence_passed']
    for mode in ('always_moving','no_restart','spontaneous'):
        assert not score(paired_features(mode=mode),parent,.05)['partial_cadence_passed']
    for variant in ('low_shift','repeat'):
        assert score(paired_features(variant=variant),parent,.05)['partial_cadence_passed']
    weak=deepcopy(paired)
    for c in CONDITIONS:
        for row in weak[c]['on']+weak[c]['off']:row['regional_cycles_rad']=[.001]*3
    assert not score(weak,parent,.05)['partial_cadence_passed']
    excessive=deepcopy(paired)
    for condition in CONDITIONS:
        for row in excessive[condition]['on']+excessive[condition]['off']:row['regional_cycles_rad']=[.03]*3
    assert not score(excessive,parent,.05)['uncued_movement_preserved']
    drift=deepcopy(paired)
    for row in drift['no_cue']['on']+drift['no_cue']['off']:row['frequency_hz']=[1.62]*3
    assert not score(drift,parent,.05)['uncued_movement_preserved']
    assert not score(paired,parent,.05)['full_stopping_qualified']


def test_features_measure_cadence_not_reduced_amplitude():
    protocol=condition_protocol('stop_start');t=np.arange(0,90.005,.005)
    f=np.where((t>=40)&(t<58),1.71,1.8);phase=2*np.pi*np.cumsum(f)*.005
    trace=dict(dense_time_s=t,dense_joints=.01*np.sin(phase)[:,None]*np.ones((1,11)),dense_voltage_mV=np.zeros((len(t),3)))
    measured=features(trace,protocol)
    assert measured['off'][0]['frequency_hz'][0]==pytest.approx(1.71,rel=.01)
    assert measured['on'][1]['frequency_hz'][0]==pytest.approx(1.8,rel=.01)
    assert min(measured['off'][0]['regional_cycles_rad'])>.019


def grouped_fixture():
    from r_elegans.data.connectome import load_connectome
    from r_elegans.brain.joint_autonomous_fit import JointFitSpec
    c=load_connectome();channels=np.ones((302,17),bool);channels[:,16]=False
    spec=JointFitSpec(channels,np.asarray(c.chemical_counts)>0,np.asarray(c.gap_counts)>0)
    origin=np.zeros(spec.size);lower=np.full(spec.size,-np.log(2));upper=-lower
    groups=parameter_groups(spec,c.neuron_ids,origin,lower,upper)
    return spec,c.neuron_ids,origin,lower,upper,groups


def test_group_decode_is_disjoint_bounded_and_preserves_motor_densities():
    spec,ids,origin,lower,upper,groups=grouped_fixture()
    assert 10<=len(groups)<=30
    all_indices=[i for g in groups for i in g['indices']];assert len(all_indices)==len(set(all_indices))
    np.testing.assert_array_equal(decode_groups(np.zeros(len(groups)),groups,origin,lower,upper),origin)
    x=decode_groups(np.linspace(-1,1,len(groups)),groups,origin,lower,upper)
    assert np.all(x>=lower) and np.all(x<=upper)
    allowed=np.zeros(len(origin),bool);allowed[all_indices]=True
    np.testing.assert_array_equal(x[~allowed],origin[~allowed])
    ci=np.nonzero(spec.channels)
    for n in ('DB01','VB01','RMDR'):
        np.testing.assert_array_equal(x[:len(ci[0])][ci[0]==ids.index(n)],0.)
    with pytest.raises(ValueError):decode_groups(np.full(len(groups),1.1),groups,origin,lower,upper)


def test_pycma_optimizer_snapshot_restores_next_population_exactly(tmp_path):
    cma=pytest.importorskip('cma')
    from scripts.run_cma_sensorimotor import optimizer_snapshot
    es=cma.CMAEvolutionStrategy([.5,-.5],.2,{'seed':7700,'popsize':8,'bounds':[-1,1],'verbose':-9,'verb_log':0})
    initial=np.linalg.norm(es.mean)
    for _ in range(12):
        points=es.ask();es.tell(points,[float(np.dot(x,x)) for x in points])
    assert np.linalg.norm(es.mean)<initial
    optimizer_snapshot(tmp_path/'state',es,0,12)
    wanted=es.ask()
    with (tmp_path/'state/optimizer.pkl').open('rb') as f:saved=pickle.load(f)
    np.random.set_state(saved['numpy_rng']);actual=saved['es'].ask()
    np.testing.assert_array_equal(wanted,actual)
    assert np.max(abs(np.asarray(actual)))<=1.


def test_coordinator_advances_only_after_checks_with_fake_simulations(tmp_path,monkeypatch):
    import concurrent.futures,json,os,sys,time
    import scripts.run_cma_sensorimotor as runner
    import scripts.compare_rhythm_worker as verifier
    import scripts.neural_timing_setup as setup_module
    import scripts.benchmark_cpu as cpu_module
    source=tmp_path/'source';(source/'control_inputs').mkdir(parents=True)
    (source/'control_inputs/continuation_parent.npz').write_bytes(b'fixture')
    out=tmp_path/'output';out.mkdir();(out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+7200,total_minutes=120)))
    monkeypatch.setattr(verifier,'verify_inputs',lambda p:dict(continuation_parent_sha256=runner.PARENT))
    monkeypatch.setattr(runner,'file_sha256',lambda p:runner.PARENT)
    monkeypatch.setattr(runner,'capture_run_record',lambda *a,**k:None)
    monkeypatch.setattr(setup_module,'setup',lambda p:dict(spec=None,model=type('M',(),{'neuron_ids':()})(),origin=np.zeros(10),lower=-np.ones(10),upper=np.ones(10)))
    monkeypatch.setattr(runner,'parameter_groups',lambda *a:[dict(name=str(i),indices=[i],lower=-.4,upper=.4) for i in range(10)])
    monkeypatch.setattr(os,'sched_getaffinity',lambda n:set(range(16)),raising=False)
    monkeypatch.setattr(cpu_module,'physical_core_cpus',lambda p:(list(range(16)),True))
    def fake(job):
        path=__import__('pathlib').Path(job['output']);path.mkdir(parents=True)
        paired=paired_features(target=job['target'],mode='always_moving' if job.get('reference') else 'controlled',variant=job['variant'])
        parent=job['parent_features'] or paired['no_cue'];m=score(paired,parent,job['target'])
        result=dict(status='completed',metrics=m,paired_features=paired,output=str(path),checkpoint_sha256='c'*64,trajectory_sha256={},elapsed_seconds=1,grid=job['grid'])
        for name in ('network.npz','coordinates.npz'): (path/name).write_bytes(b'fixture')
        (path/'request.json').write_text(json.dumps(job));(path/'summary.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(runner,'evaluate',fake)
    class Pool:
        def __init__(self,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def submit(self,fn,j):
            f=concurrent.futures.Future();f.set_result(fn(j));return f
    monkeypatch.setattr(concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(sys,'argv',['runner','--output-dir',str(out),'--source-run',str(source)])
    runner.main();s=json.loads((out/'summary.json').read_text())
    assert s['completed_generations']==5
    assert s['curriculum_rungs_passed_training_checks']==[.05,.1,.2,.35,.5]
    assert s['validation_complete'] and s['final_grid_partial_cadence_pass']
    assert not s['full_stopping_qualified'] and s['swimming_blocked']
    assert (out/'selected/final/network.npz').exists()


def test_persistent_worker_reuses_rollouts_and_supplies_only_environmental_cues(tmp_path,monkeypatch):
    import scripts.cma_sensorimotor_worker as worker
    import scripts.compare_rhythm_worker as verifier
    import r_elegans.brain.conductance_network as network
    factory_calls=[]
    def factory(dt,brain,**kwargs):
        assert 'diagnostic_membrane' not in kwargs
        world=kwargs['world_override'];levels=np.asarray(world.concentration_um)
        stop=levels[:,2].max()>50000;start=levels[:,0].max()>0;factory_calls.append((stop,start))
        def rollout(params,times,gains):
            t=np.arange(0,90.005,.005);slow=(t>=40)&((t<58) if start else True) if stop else np.zeros(len(t),bool)
            frequency=np.where(slow,1.71,1.8);phase=np.cumsum(frequency)*2*np.pi*.005
            return dict(dense_time_s=t,dense_joints=.01*np.sin(phase)[:,None]*np.ones((1,11)),dense_voltage_mV=np.zeros((len(t),3)))
        return rollout
    monkeypatch.setattr(worker,'_CACHE',{})
    monkeypatch.setattr(worker,'_DATA',dict(origin=np.zeros(1),lower=-np.ones(1),upper=np.ones(1),groups=[dict(indices=[0],lower=-.4,upper=.4)],decode=lambda x:x,fixed=None,config=None,factory=factory,gains=np.ones(3),root=tmp_path))
    monkeypatch.setattr(network,'save_network',lambda path,*a:path.write_bytes(b'fixture'))
    monkeypatch.setattr(verifier,'verify_inputs',lambda p:None)
    import time
    until=time.monotonic()+60
    for i in range(2):
        result=worker.evaluate(dict(output=str(tmp_path/f'eval-{i}'),z=[.1],target=.05,until=until))
        assert result['status']=='completed' and result['metrics']['partial_cadence_passed']
    assert len(factory_calls)==4
