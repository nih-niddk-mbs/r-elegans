from copy import deepcopy
import numpy as np
import pytest
from r_elegans.avb_training import protocol,features,score,membrane_probe,parameter_groups


def trace(inhibited=False,no_recovery=False,variant='ordinary'):
    duration,intervals,_=protocol(variant);t=np.arange(0,duration,.01)
    off=np.zeros(len(t),bool)
    for a,b in intervals:off|=(t>=a)&(t<b)
    if no_recovery:off|=t>=intervals[0][0]
    ids=('AVBL','AVBR','DB01','VB01','RMDR')
    gain=np.where(off&inhibited,0.,1.)
    v=np.tile(np.array([-30.,-30.,-40.,-40.,-40.]),(len(t),1))
    v[:,2:]+=gain[:,None]*10*np.sin(2*np.pi*1.8*t[:,None])
    v[:,:2]-=(off&inhibited)[:,None]*30
    return dict(dense_time_s=t,dense_joints=gain[:,None]*.01*np.sin(2*np.pi*1.8*t[:,None])*np.ones((1,11)),dense_voltage_mV=v),ids


def test_requires_preserved_gait_selective_suppression_neural_effect_and_restart():
    for variant in ('ordinary','repeat','shift-weak'):
        a,ids=trace(variant=variant);b,_=trace(True,variant=variant)
        sham=features(a,ids,variant);off=features(b,ids,variant)
        result=score(sham,off,sham)
        assert result['avb_component_passed']
        assert not result['sensory_start_stop_qualified'] and not result['swimming_qualified']
        assert not score(sham,sham,sham)['avb_component_passed']
    b,ids=trace(True,no_recovery=True)
    assert not score(sham,features(b,ids),sham)['avb_component_passed']
    a,ids=trace();sham=features(a,ids)
    dead=deepcopy(sham)
    for row in dead['active']+dead['off']:
        row.update(regional_cycles_rad=[0.]*3,raw_range_rad=0.,motor_rms_mV=0.,frequency_hz=[None]*3)
    assert not score(dead,dead,sham)['avb_component_passed']
    b,ids=trace(True);off=features(b,ids)
    off['off'][0]['motor_rms_mV']=sham['off'][0]['motor_rms_mV']
    assert not score(sham,off,sham)['avb_component_passed']


def test_loss_improves_continuously_without_waiting_for_frequency_bins():
    a,ids=trace();sham=features(a,ids)
    values=[]
    for factor in (1.,.99,.9,.5):
        off=deepcopy(sham)
        off['off'][0]['avb_mean_mV']-=30
        off['off'][0]['raw_range_rad']*=factor
        off['off'][0]['motor_rms_mV']*=factor
        values.append(score(sham,off,sham)['loss'])
    assert all(a>b for a,b in zip(values,values[1:]))


def test_probe_targets_only_avb_and_releases_without_voltage_freeze():
    ids=('AVBL','DB01','AVBR','RIS');probe=membrane_probe(ids,'repeat')
    for t in (0.,29.99,45.,70.,90.,119.):
        assert not np.asarray(probe(t).conductance_nS).any()
    for t in (30.,44.,75.,89.):
        result=probe(t)
        np.testing.assert_allclose(result.conductance_nS,[.1,0.,.1,0.])
    with pytest.raises(ValueError):protocol('unknown')


def test_grouped_fit_includes_motor_channels_without_leak_or_overlap():
    from test_cma_curriculum import grouped_fixture
    from r_elegans.cma_curriculum import decode_groups
    spec,ids,origin,lower,upper,_=grouped_fixture()
    groups=parameter_groups(spec,ids,origin,lower,upper)
    all_indices=[i for g in groups for i in g['indices']]
    assert len(all_indices)==len(set(all_indices))
    ci=np.nonzero(spec.channels);x=decode_groups(np.ones(len(groups)),groups,origin,lower,upper)
    for name in ('DB01','VB01','RMDR','AVBL','AVBR'):
        assert np.any(x[:len(ci[0])][ci[0]==ids.index(name)]!=0)
    assert np.all(x<=upper) and np.all(x>=lower)
    np.testing.assert_array_equal(decode_groups(np.zeros(len(groups)),groups,origin,lower,upper),origin)


def test_worker_keeps_models_and_metrics_without_intermediate_trace_files(tmp_path,monkeypatch):
    import time
    import scripts.avb_training_worker as worker
    import scripts.cma_sensorimotor_worker as common
    import scripts.compare_rhythm_worker as verifier
    import r_elegans.brain.conductance_network as network
    a,ids=trace();b,_=trace(True)
    calls=[]
    def factory(dt,brain,**kwargs):
        inhibited=kwargs['diagnostic_membrane'] is not None;calls.append(inhibited)
        return lambda *args:b if inhibited else a
    monkeypatch.setattr(common,'_DATA',dict(origin=np.zeros(1),lower=-np.ones(1),upper=np.ones(1),
        groups=[dict(indices=[0],lower=-.4,upper=.4)],decode=lambda x:x,fixed=None,config=None,
        factory=factory,gains=np.ones(3),root=tmp_path/'inputs',model=type('M',(),{'neuron_ids':ids})()))
    monkeypatch.setattr(worker,'_CACHE',{})
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda p:type('D',(),{'free':1024**4})())
    monkeypatch.setattr(network,'save_network',lambda path,*args:path.write_bytes(b'model'))
    monkeypatch.setattr(verifier,'verify_inputs',lambda p:None)
    result=worker.evaluate(dict(z=[0.],output=str(tmp_path/'candidate'),until=time.monotonic()+60))
    assert result['status']=='completed' and result['metrics']['avb_component_passed']
    assert calls==[False,True]
    assert (tmp_path/'candidate/network.npz').exists() and (tmp_path/'candidate/coordinates.npz').exists()
    assert not (tmp_path/'candidate/sham.npz').exists()
    assert not (tmp_path/'candidate/inhibited.npz').exists()


def test_coordinator_preserves_parent_state_and_validates_exact_selected_model(tmp_path,monkeypatch):
    import concurrent.futures,json,os,sys,time
    import scripts.run_avb_training as runner
    import scripts.compare_rhythm_worker as verifier
    import scripts.neural_timing_setup as setup_module
    import scripts.benchmark_cpu as cpu_module
    source=tmp_path/'source';(source/'control_inputs').mkdir(parents=True)
    (source/'control_inputs/continuation_parent.npz').write_bytes(b'fixture')
    out=tmp_path/'output';out.mkdir()
    (out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+28800,total_minutes=480)))
    monkeypatch.setattr(verifier,'verify_inputs',lambda p:dict(continuation_parent_sha256=runner.PARENT))
    monkeypatch.setattr(runner,'file_sha256',lambda p:runner.PARENT)
    monkeypatch.setattr(runner,'capture_run_record',lambda *args,**kwargs:None)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:type('D',(),{'free':1024**4})())
    monkeypatch.setattr(setup_module,'setup',lambda p:dict(spec=None,model=type('M',(),{'neuron_ids':()})(),origin=np.zeros(10),lower=-np.ones(10),upper=np.ones(10)))
    monkeypatch.setattr(runner,'parameter_groups',lambda *args:[dict(name=str(i),indices=[i],lower=-.4,upper=.4) for i in range(10)])
    monkeypatch.setattr(os,'sched_getaffinity',lambda n:set(range(16)),raising=False)
    monkeypatch.setattr(cpu_module,'physical_core_cpus',lambda p:(list(range(16)),True))
    def fake(job):
        path=__import__('pathlib').Path(job['output']);path.mkdir(parents=True)
        a,ids=trace(variant=job['variant']);b,_=trace(True,variant=job['variant'])
        sham=features(a,ids,job['variant']);off=features(a if job['reference'] else b,ids,job['variant'])
        result=dict(status='completed',metrics=score(sham,off,sham),features=dict(sham=sham,inhibited=off),output=str(path),
                    checkpoint_sha256='c'*64,elapsed_seconds=1,grid=job['grid'],variant=job['variant'])
        for name in ('network.npz','coordinates.npz'):(path/name).write_bytes(b'fixture')
        (path/'request.json').write_text(json.dumps(job));(path/'summary.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(runner,'evaluate',fake)
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,j):
            f=concurrent.futures.Future();f.set_result(fn(j));return f
    monkeypatch.setattr(concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(sys,'argv',['runner','--output-dir',str(out),'--source-run',str(source)])
    runner.main();s=json.loads((out/'summary.json').read_text())
    assert s['completed_generations']==1
    assert s['validation_complete'] and s['avb_component_qualified']
    assert not s['sensory_start_stop_qualified'] and not s['swimming_qualified']
    assert (out/'selected/network.npz').exists() and (out/'optimizer/final/optimizer.pkl').exists()
