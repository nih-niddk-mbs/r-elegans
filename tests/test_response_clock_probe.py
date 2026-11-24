import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from r_elegans.brain.response_clock_probe import clock_counts,make_replay,response_differences,CLOCKS
from scripts import run_response_clock_diagnostic as runner


def test_clocks_are_aligned_and_nondividing_clocks_refused():
    for dt,step in CLOCKS.values():
        sub,block,frames=clock_counts(dt,step)
        assert np.isclose(sub*step,dt*1000) and np.isclose(block*dt,.001) and frames==12000
    with pytest.raises(ValueError):clock_counts(.0003,.03125)


def test_chunked_export_preserves_each_outer_update_and_window_mean(monkeypatch):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain import conductance_network as neural,extrasynaptic as extra
    from r_elegans.brain.peptide_state_replay import FullState
    def advance(s,p,f,dt,sub,c,membrane_input):
        v=s.voltage_mV+dt*(1+membrane_input.conductance_nS)
        return s._replace(voltage_mV=v,indicator=s.indicator+dt*v,calcium_uM=s.calcium_uM+dt)
    monkeypatch.setattr(neural,'advance_network',advance)
    monkeypatch.setattr(extra,'peptide_membrane_input',lambda state,e,i,u:SimpleNamespace(conductance_nS=state,weighted_reversal_pA=jnp.zeros_like(state)))
    monkeypatch.setattr(extra,'peptide_level_step',lambda level,v,f,dt,tau:level+dt)
    start=FullState(neural.ConductanceNetworkState(jnp.zeros((4,2)),jnp.zeros((4,2,35)),jnp.zeros((4,2)),jnp.zeros((4,2))),jnp.zeros((4,2)))
    gs=jnp.asarray([[.03,0.],[0.,0.],[0.,0.]])
    replay=make_replay(SimpleNamespace(synaptic_unit_nS=1.),[0,1],.00025,.0078125,seconds=.004,stimulus_seconds=.002,window=(.001,.003))
    pp=extra.PeptideParams(jnp.zeros(1),jnp.zeros(1),jnp.array(0.))
    means,final,trace=jax.tree.map(np.asarray,replay(None,None,pp,None,None,start,gs))
    v=np.zeros((3,2));indicator=np.zeros_like(v);peptide=np.zeros_like(v);total=np.zeros_like(v);exports=[]
    for k in range(16):
        v+=.00025*(1+peptide+np.asarray(gs)*(k<8));indicator+=.00025*v;peptide+=.00025
        if 4<=k<12:total+=indicator
        if k%4==3:exports.append(v.copy())
    np.testing.assert_allclose(means[0],total/8,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(final.neural.voltage_mV[0],v,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(trace[0][0],np.stack(exports,axis=1),rtol=1e-12,atol=1e-15)
    np.testing.assert_array_equal(means[:,1],means[:,2])


def test_descriptive_metrics_keep_phase_and_mean_separate():
    a=np.zeros((4,3,2));b=a.copy();a[:,0,0]=[1,-1,1,-1]
    result=response_differences(a,b)
    assert result['phasewise_max']==1 and result['phase_mean_max']==0 and 'passed' not in result


def test_case_controls_and_response_arithmetic():
    data=dict(means=np.zeros((4,5,2)),response=np.zeros((4,3,2)),voltage_min=np.zeros((4,5,10)),voltage_max=np.ones((4,5,10)))
    for k in ('voltage_mV','calcium_uM','indicator','peptide'):data[k]=np.zeros((4,5,10,2))
    result=runner.case_metrics(data,dict(voltage_mV=np.zeros((4,5,2))))
    assert all(result['checks'].values()) and result['response_from_means_error']==0
    data['indicator'][0,-1,0,0]=1
    assert not runner.case_metrics(data,{})['checks']['duplicate_sham_passed']


def test_readonly_modes_do_not_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'jobs',lambda:[])
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:pytest.fail('Read-only launch'))
    for args in (['--status'],['--output-dir',str(tmp_path),'--report']):
        monkeypatch.setattr(sys,'argv',['probe']+args);runner.main()
    assert not list(tmp_path.iterdir())


def test_existing_job_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'jobs',lambda:[dict(pid=123)])
    monkeypatch.setattr(sys,'argv',['probe','--output-dir',str(tmp_path/'new')])
    with pytest.raises(SystemExit):runner.main()
    assert not (tmp_path/'new').exists()


def test_timeout_audits_partial_execution(tmp_path,monkeypatch):
    from scripts import total_job_budget
    monkeypatch.setattr(runner,'jobs',lambda:[]);monkeypatch.setattr(runner,'BASE',tmp_path)
    monkeypatch.setattr(total_job_budget,'run_group',lambda *a:(124,True))
    root=tmp_path/'new';monkeypatch.setattr(sys,'argv',['probe','--output-dir',str(root)])
    assert runner.main()==2
    receipt=json.loads((root/'budget_receipt.json').read_text())
    assert receipt['timed_out'] and receipt['audit_complete'] and not receipt['all_steps_finished']


def test_preparation_pins_models_and_preserves_saved_state_choices(tmp_path,monkeypatch):
    import shutil
    from r_elegans.brain.peptide_checkpoint import bundle_model
    repo=Path(__file__).resolve().parents[1];source=tmp_path/'source';source.mkdir();out=tmp_path/'out';out.mkdir()
    (source/'audit.json').write_text(json.dumps(dict(integrity_passed=True,execution_complete=True)))
    (source/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True)))
    origin=repo/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz'
    asset=runner.arrays(repo/'r_elegans/assets/extrasynaptic_v1.npz');models={}
    for arm in ('network','peptide'):
        folder=source/arm/'fold-0';folder.mkdir(parents=True);paths=[]
        for level in range(3):
            path=folder/f'level-{level}';path.mkdir();paths.append(str(path))
            shutil.copy2(origin,path/'network.npz')
            state=dict(start_voltage_mV=np.full((4,302),-35.+level),start_gates=np.full((4,302,35),.5),
                start_calcium_uM=np.full((4,302),.05),start_indicator=np.zeros((4,302)),start_peptide=np.full((4,302),.2+level*.1))
            targets=runner.TARGETS if level<2 else []
            np.savez_compressed(path/'prediction.npz',**state,response=np.zeros((4,len(targets),302)))
            meta=dict(level=level,targets=targets,phase=dict(indices=[1,2,3,4]),
                sha256={str(path/n):runner.file_sha256(path/n) for n in ('prediction.npz','network.npz')})
            (path/'metrics.json').write_text(json.dumps(meta))
        model=folder/'selected-model.npz'
        bundle_model(origin,Path(paths[2])/'prediction.npz',model,enabled=arm=='peptide',asset=asset,asset_sha256='fixture',network_sha256=runner.file_sha256(origin))
        models[arm]=(0,runner.file_sha256(model))
        (folder/'summary.json').write_text(json.dumps(dict(model_sha256=models[arm][1],prediction_paths=paths[:2],quarter_path=paths[2])))
    monkeypatch.setattr(runner,'MODELS',models);monkeypatch.setattr(runner,'physical_cpus',lambda:list(range(8)))
    runner.prepare(out,source)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['models']['network']['cpus']==[0,1,2,3]
    assert manifest['models']['peptide']['cpus']==[4,5,6,7]
    for arm in models:
        for level in range(3):
            np.testing.assert_array_equal(runner.arrays(out/arm/f'start-{level}.npz')['voltage_mV'],np.full((4,302),-35.+level))
        assert manifest['models'][arm]['references']['2']['targets']==[]
        assert runner.file_sha256(out/arm/'model.npz')==models[arm][1]
