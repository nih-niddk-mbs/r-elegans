from copy import deepcopy
import json,sys,time
import numpy as np
import pytest
from r_elegans.smd_control import SMD,cases,protocol,membrane_probe,compare,choose


def traces(variant='nominal',mode='controlled'):
    duration,intervals,_=protocol(variant);t=np.arange(0,duration+.005,.01)
    active=np.zeros(len(t),bool)
    for a,b in intervals:active|=(t>=a)&(t<b)
    if mode=='no_recovery':active|=t>=intervals[0][0]
    ids=('AVBL','DB01')+SMD
    v=-35.+5*np.sin(2*np.pi*1.8*t[:,None])*np.ones((1,len(ids)))
    q=.01*np.sin(2*np.pi*1.8*t[:,None])*np.ones((1,11))
    muscle=.5+.1*np.sin(2*np.pi*1.8*t[:,None])*np.ones((1,95))
    sham=dict(dense_time_s=t,dense_voltage_mV=v,dense_joints=q,dense_muscles=muscle)
    test={k:x.copy() for k,x in sham.items()}
    if mode!='body_only':test['dense_voltage_mV'][active,2:]=-70.
    if mode!='neural_only':
        test['dense_joints'][active]*=.1
        test['dense_muscles'][active]=.5+.1*(test['dense_muscles'][active]-.5)
    if mode=='pre_changed':test['dense_joints'][t<20]*=.8
    return test,sham,ids


def test_neural_access_requires_actual_neural_and_body_suppression_and_recovery():
    case=cases()[4]
    for variant in ('nominal','shift','repeat','long-hold'):
        test,sham,ids=traces(variant)
        result=compare(test,sham,ids,case,variant)
        assert result['reversible_neural_target']
        assert not result['strict_body_rest'] and not result['learned_stopping_qualified']
    for mode in ('no_recovery','body_only','neural_only','pre_changed'):
        test,sham,ids=traces(mode=mode)
        assert not compare(test,sham,ids,case)['reversible_neural_target']
    test,sham,ids=traces();test['dense_time_s']+=.001
    with pytest.raises(ValueError):compare(test,sham,ids,case)


def test_smd_only_probes_have_explicit_strength_and_timing():
    import jax
    all_cases=cases();assert len(all_cases)==11
    assert len({c['name'] for c in all_cases})==11
    ids=('AVBL','DB01')+SMD
    for c in all_cases:
        probe=jax.jit(membrane_probe(ids,c,'repeat'))
        for t in (0.,29.9,40.,75.,90.,119.):
            assert not np.asarray(probe(t).conductance_nS).any()
        for t in (30.,35.,80.,85.):
            expected=np.array([c['conductance_nS'] if n in c['targets'] else 0. for n in ids])
            result=probe(t);np.testing.assert_allclose(result.conductance_nS,expected)
            np.testing.assert_allclose(result.weighted_reversal_pA,-80*expected)
    with pytest.raises(ValueError):membrane_probe(ids,dict(targets=['AVBL'],conductance_nS=.1),'nominal')
    with pytest.raises(ValueError):membrane_probe(ids,dict(targets=list(SMD),conductance_nS=1.),'nominal')


def test_selection_preserves_recovery_and_reports_no_confirmed_access():
    def row(name,body,recover,access=True):
        return dict(case=dict(name=name,conductance_nS=.1),metrics=dict(pre_match=True,safe=True,
            active_and_recovery_preserved=recover,off_windows=[dict(body_range_ratio=body,inhibition_confirmed=access)]))
    assert choose([row('trapped',0.,False),row('reversible',.15,True)])['name']=='reversible'
    assert choose([row('no_inhibition',0.,True,False)]) is None


def test_smd_watchdog_dispatch_and_budget_receipt(tmp_path,monkeypatch):
    import scripts.total_job_budget as runner
    out=tmp_path/'output';seen=[]
    def run(command,seconds):
        seen.append(command);(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,validation_complete=True)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','smd-control','--output-dir',str(out),'--source-run',str(tmp_path/'source'),'--total-minutes','240'])
    assert runner.main()==0 and seen[0][3]=='scripts.run_smd_control'
    assert json.loads((out/'deadline.json').read_text())['experiment']=='E082'
    assert json.loads((out/'budget_receipt.json').read_text())['total_minutes']==240


def test_worker_uses_live_membrane_probe_and_retains_source_checked_trace(tmp_path,monkeypatch):
    import scripts.smd_control_worker as worker
    import scripts.neural_timing_setup as setup_module
    import scripts.compare_rhythm_worker as verifier
    import r_elegans.brain.conductance_network as network
    import r_elegans.rhythm_comparison as reconstruction
    from scripts.experiment_records import file_sha256
    test,sham,ids=traces();root=tmp_path/'inputs';root.mkdir();out=tmp_path/'trial'
    reference=tmp_path/'reference.npz';np.savez_compressed(reference,**sham)
    (root/'reference.json').write_text(json.dumps(dict(path=str(reference),sha256=file_sha256(reference))))
    (root/'diagnostic_model.npz').write_bytes(b'fixture')
    np.savez_compressed(root/'diagnostic_coordinates.npz',offsets=np.zeros(1))
    seen=[]
    def factory(dt,brain,**kwargs):
        assert 'diagnostic_nmj_release' not in kwargs
        probe=kwargs['diagnostic_membrane'];assert probe is not None
        seen.append(np.asarray(probe(35.).conductance_nS))
        return lambda *args:test
    params=(np.zeros(1),)
    monkeypatch.setattr(setup_module,'setup',lambda root:dict(decode=lambda x:params,fixed=None,config=None,model=type('M',(),{'neuron_ids':ids})(),factory=factory,gains=np.ones(3)))
    monkeypatch.setattr(verifier,'verify_inputs',lambda root:None)
    monkeypatch.setattr(network,'load_network',lambda path:(params,None,None))
    monkeypatch.setattr(reconstruction,'assert_fixed_equal',lambda *args:None)
    monkeypatch.setattr(worker,'capture_run_record',lambda *args:None)
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda path:type('D',(),{'free':1024**4})())
    monkeypatch.setattr(sys,'argv',['worker','--inputs',str(root),'--output-dir',str(out),'--case','SMD-0.1','--until',str(time.monotonic()+60)])
    worker.main();r=json.loads((out/'summary.json').read_text())
    assert r['completed'] and r['no_training'] and not r['learned_stopping_qualified']
    np.testing.assert_allclose(seen[0],[0,0,.1,.1,.1,.1])
    assert r['source_reference_errors']['dense_voltage_mV']==0
    assert file_sha256(out/'trajectory.npz')==r['trajectory_sha256']
