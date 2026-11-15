import numpy as np
import pytest
from scipy.special import expit
from r_elegans.avb_audit import incoming_currents,cell_rows,analyze_pair,load_saved_network,density_group_members


def network():
    ids=['AVBL','AVBR','DB01','VB01','RMDL','ASHL'];n=len(ids)
    chemical=np.zeros((n,n));gap=np.zeros((n,n))
    chemical[2,0]=.2;gap[3,1]=gap[1,3]=.3
    return dict(ids=ids,chemical=chemical,gap=gap,reversal=np.zeros(n),threshold=np.full(n,-40.),slope=np.full(n,5.),
                channel_g=np.ones((n,17)),bias=np.zeros(n),leak_reversal=np.full(n,-60.))


def test_current_orientation_sign_and_individual_avb_direct_influence():
    d=network();v=np.full((100,6),-40.);w=v.copy();w[:,0:2]=-70
    r=expit((v-d['threshold'])/d['slope'])
    c,g=incoming_currents(v,r,2,d)
    assert c[0,0]==pytest.approx(4.)
    assert np.count_nonzero(c)==100 and not g.any()
    rows={r['neuron']:r for r in cell_rows(v,w,d)}
    assert 'ASHL' not in rows  # AS sensory neurons are not numbered AS motor neurons.
    assert rows['DB01']['avb_L1_fraction']==pytest.approx(1.)
    assert rows['DB01']['avb_direct_chemical_delta_mean_pA']<0
    assert rows['VB01']['avb_direct_gap_delta_mean_pA']==pytest.approx(-9.)
    assert rows['RMDL']['avb_direct_total_delta_rms_pA']==0.
    assert rows['DB01']['top_incoming_sham'][0]['neuron']=='AVBL'
    assert rows['DB01']['voltage_std_ratio'] is None


def test_decomposition_separates_presynaptic_signal_and_postsynaptic_feedback():
    d=network();v=np.full((100,6),-40.);w=v.copy();w[:,2]=-30.
    rows={r['neuron']:r for r in cell_rows(v,w,d)}
    assert rows['DB01']['avb_direct_total_delta_rms_pA']==0.
    assert rows['DB01']['actual_all_synapse_delta_rms_pA']>0.


def test_analysis_requires_matching_clocks_and_precue_state():
    d=network();t=np.arange(0,60.002,.002)
    v=np.full((len(t),6),-40.);v[:,2:]+=np.sin(2*np.pi*1.8*t[:,None]);w=v.copy();w[(t>=30)&(t<40),:2]-=30
    a=dict(dense_time_s=t,dense_voltage_mV=v);b=dict(dense_time_s=t,dense_voltage_mV=w)
    report=analyze_pair(a,b,d)
    assert report['windows']['off']['sample_stride']==5
    assert report['pre_intervention_max_voltage_difference_mV']==0.
    b['dense_time_s']=t+.001
    with pytest.raises(ValueError,match='clocks'):analyze_pair(a,b,d)
    b['dense_time_s']=t;b['dense_voltage_mV']=w.copy();b['dense_voltage_mV'][0,0]+=1
    with pytest.raises(ValueError,match='Pre-intervention'):analyze_pair(a,b,d)


def test_numpy_loader_matches_runtime_synapse_formula(tmp_path):
    import jax
    jax.config.update('jax_enable_x64',True)
    from r_elegans.brain.conductance_network import load_network,network_synapses
    # Use a bundled real canonical checkpoint; no neural simulation.
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'r_elegans/assets/autonomous_wiggle_v1/network.npz'
    params,fixed,config=load_network(path)
    params,fixed=jax.tree.map(lambda x:x.astype('float64') if x.dtype.kind=='f' else x,(params,fixed))
    c,g=network_synapses(params,fixed,config);d=load_saved_network(path)
    np.testing.assert_allclose(d['chemical'],c,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(d['gap'],g,rtol=1e-12,atol=1e-14)


def test_inventory_uses_actual_saved_indices_to_expose_prefix_overreach():
    d=network()
    rows=density_group_members([dict(name='density:AS:inward',indices=[5*16+6])],d)
    assert rows[0]['cells']==['ASHL']


def test_audit_cli_verifies_saved_evidence_and_writes_small_reports(tmp_path,monkeypatch):
    import json,sys
    from scripts import audit_avb_influence as runner
    from scripts.experiment_records import file_sha256
    d=network();n=len(d['ids']);run=tmp_path/'run';run.mkdir()
    t=np.arange(0,60.01,.01);v=np.full((len(t),n),-40.);w=v.copy();w[(t>=30)&(t<40),:2]-=30
    records=[]
    for name in ('parent','selected'):
        folder=run/name;folder.mkdir()
        path=folder/'network.npz'
        np.savez_compressed(path,schema_version=1,neuron_ids=d['ids'],config_json=json.dumps(dict(synaptic_unit_nS=.03)),
            params_raw_chemical=np.zeros((n,n)),params_raw_gap=np.zeros((n,n)),params_log_channel_scale=np.zeros((n,17)),
            fixed_chemical_mask=d['chemical']>0,fixed_gap_mask=d['gap']>0,
            fixed_synapse_reversal_mV=d['reversal'],fixed_release_threshold_mV=d['threshold'],fixed_release_slope_mV=d['slope'],
            cells_conductances_nS=d['channel_g'],fixed_bias_pA=d['bias'],cells_leak_reversal_mV=d['leak_reversal'])
        hashes={}
        for c,voltage in [('sham',v),('inhibited',w)]:
            np.savez_compressed(folder/(c+'.npz'),dense_time_s=t,dense_voltage_mV=voltage)
            hashes[c]=file_sha256(folder/(c+'.npz'))
        records.append(dict(output=str(folder),checkpoint_sha256=file_sha256(path),trajectory_sha256=hashes,variant='ordinary',grid='ordinary'))
    (run/'parameter_groups.json').write_text('[]')
    (run/'summary.json').write_text(json.dumps(dict(experiment='E078',validation_complete=True,baseline=records[0],selected=records[1],validation=[records[1]])))
    monkeypatch.setattr(runner,'capture_run_record',lambda *args,**kwargs:None)
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['audit','--run',str(run),'--output-dir',str(out)])
    runner.main()
    summary=json.loads((out/'summary.json').read_text())
    assert summary['no_training'] and summary['no_new_simulation']
    assert (out/'parent.json').exists() and (out/'selected.json').exists()
    assert len(json.loads((out/'evidence_hashes.json').read_text()))==8
    assert not list(out.glob('*.npz'))
