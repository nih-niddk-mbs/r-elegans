import numpy as np
from scipy.special import expit
from scripts.sensory_motion_details import edge_summary,bound_rows


def test_release_change_and_actual_current_change_keep_driving_force_separate():
    network=dict(ids=['AIYL','RIAL'],chemical=np.array([[0.,0.],[1.,0.]]),gap=np.array([[0.,.2],[.2,0.]]),
        reversal=np.array([0.,-85.]),threshold=np.array([-40.,-40.]),slope=np.array([5.,5.]))
    a=np.tile([-40.,-50.],(4,1));b=np.tile([-35.,-40.],(4,1))
    row=edge_summary(a,b,network,1,0)
    np.testing.assert_allclose(row['release_only_delta_mean_pA'],(expit(1)-.5)*50)
    np.testing.assert_allclose(row['actual_chemical_delta_mean_pA'],expit(1)*40-.5*50)
    np.testing.assert_allclose(row['presynaptic_gap_delta_mean_pA'],1.)
    np.testing.assert_allclose(row['actual_total_delta_rms_pA'],abs(expit(1)*40-.5*50-1.))
    assert row['release_only_delta_mean_pA']>row['actual_chemical_delta_mean_pA']
    network['reversal'][0]=-85.
    assert edge_summary(a,b,network,1,0)['release_only_delta_mean_pA']<0


def test_bound_report_distinguishes_original_exception_from_declared_cap_room():
    origin=np.array([0.,.7]);offsets=np.array([.4,.8]);lower=np.array([-.69,-.69]);upper=np.array([.69,.69])
    rows=bound_rows([dict(name='density',indices=[0]),dict(name='AIYR->AIYL',indices=[1])],
        [1.,.5],origin,offsets,lower,upper,1)
    assert rows[0]['near_local_bound'] and rows[0]['original_upper_exceptions']==0
    assert rows[1]['original_upper_exceptions']==1
    assert rows[1]['minimum_original_upper_room_log']<0
    assert rows[1]['minimum_declared_upper_room_log']>0
    assert not rows[1]['near_local_bound']


def test_saved_report_runs_on_retained_traces_without_changing_source(tmp_path,monkeypatch):
    import json,shutil
    from pathlib import Path
    from types import SimpleNamespace
    from scipy.special import logit
    import scripts.sensory_motion_details as details
    from scripts.experiment_records import file_sha256
    from r_elegans.sensory_motion import features,parameter_groups
    from r_elegans.relay_cap_experiments import edge_coordinate
    run=tmp_path/'run';inputs=run/'inputs';inputs.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    asset=Path('r_elegans/assets/autonomous_wiggle_v1/network.npz')
    shutil.copyfile(asset,inputs/'motion_parent.npz')
    with np.load(asset,allow_pickle=False) as z:
        ids=tuple(z['neuron_ids']);channels=z['cells_conductances_nS']>0;channels[:,16]=False;chemical=z['fixed_chemical_mask'].astype(bool)
        threshold=float(z['fixed_release_threshold_mV'][ids.index('AIYL')]);slope=float(z['fixed_release_slope_mV'][ids.index('AIYL')])
    spec=SimpleNamespace(channels=channels,chemical=chemical);size=int(channels.sum()+chemical.sum())
    origin=np.zeros(size);lower=np.full(size,-np.log(2.));upper=-lower;edge=edge_coordinate(spec,ids)
    groups=parameter_groups(spec,ids,origin,lower,upper)
    np.savez(inputs/'cap_prior.npz',origin=origin,lower=lower,upper=upper,edge=edge)
    write(inputs/'cap_groups.json',dict(aiyr_2p5=groups))
    write(inputs/'manifest.json',dict(sha256={p.name:file_sha256(p) for p in inputs.iterdir()}))
    t=np.arange(0,83.001,.02);traces={};pairs={};nominal={}
    for condition in ('no_cue','cue'):
        r=np.full(len(t),.95);gain=np.ones(len(t))
        if condition=='cue':
            for a,b in ((41.,46.),(54.,59.)):
                r[(t>=a)&(t<b)]-=.0001;gain[(t>=a)&(t<b)]=.99
        v=np.tile((-40+np.sin(2*np.pi*1.8*t))[:,None],(1,len(ids)))
        v[:,ids.index('AIYL')]=threshold+slope*logit(r)
        traces[condition]=dict(dense_time_s=t,dense_voltage_mV=v,dense_joints=.01*gain[:,None]*np.sin(2*np.pi*1.8*t[:,None])*np.ones((1,11)))
        pairs[condition]=features(traces[condition],ids,'redundant',threshold,slope)
        nominal[condition]=features(traces[condition],ids,'nominal',threshold,slope)
    write(run/'parent_features.json',pairs['no_cue'])
    folder=run/'evaluations/selected';folder.mkdir(parents=True)
    shutil.copyfile(asset,folder/'network.npz');np.savez(folder/'coordinates.npz',offsets=origin,z=np.zeros(len(groups)))
    write(folder/'summary.json',dict(paired_features=pairs,checkpoint_sha256=file_sha256(asset)))
    initial=run/'evaluations/parent-redundant';initial.mkdir();write(initial/'summary.json',dict(paired_features=pairs))
    write(run/'generation_history.json',[dict(generation=1,selected_evaluation=str(folder))])
    final=run/'evaluations/final-00';final.mkdir();shutil.copyfile(asset,final/'network.npz')
    hashes={}
    for condition,trace in traces.items():
        np.savez_compressed(final/(condition+'.npz'),**trace);hashes[condition]=file_sha256(final/(condition+'.npz'))
    row=dict(output=str(final),paired_features=nominal,trajectory_sha256=hashes)
    sha=file_sha256(asset)
    write(run/'summary.json',dict(experiment='E093',execution_valid=True,validation_complete=True,target=.01,
        panels=[dict(training_output=str(folder),checkpoint_sha256=sha,validation=[row])]))
    write(run/'configuration.json',{})
    write(run/'budget_receipt.json',dict(all_steps_finished=True))
    audit=run.with_name('run-audit');audit.mkdir();write(audit/'summary.json',dict(audit_complete=True,integrity_passed=True))
    paths=[folder/'network.npz',folder/'coordinates.npz',final/'network.npz',final/'no_cue.npz',final/'cue.npz']
    write(audit/'evidence.json',[dict(path=str(p),sha256=file_sha256(p)) for p in paths])
    before={str(p):file_sha256(p) for p in run.rglob('*') if p.is_file()}
    monkeypatch.setattr(details,'capture_run_record',lambda *a,**k:None)
    out=tmp_path/'derived';result=details.analyze(run,out)
    assert result['loss_reduction_percent']==0 and result['changed_coordinates']==0 and result['frozen_fields_unchanged']
    assert len(result['signal_windows'])==4
    assert result['signal_windows'][2]['AIYL_to_RIAL']['release_only_delta_mean_pA']!=0
    assert (out/'evidence.json').is_file()
    assert before=={str(p):file_sha256(p) for p in run.rglob('*') if p.is_file()}
