import concurrent.futures,json,sys,time
from pathlib import Path
import numpy as np
import pytest
from r_elegans.node_probe import cases,membrane_probe,retains_trace,predict_at,CALIBRATION_GROUPS,CALIBRATION_DOSES,TARGET,reduction_percent,reaches_target,minimum_doses
from r_elegans.sensory_motion import features,score
from test_sensory_motion import data
from test_rial_sensory import paired


def test_calibration_profile_adds_low_doses_aiy_and_grid_controls_without_changing_e096():
    c=cases('e097');assert len(c)==len(CALIBRATION_GROUPS)*2*len(CALIBRATION_DOSES)+4==52 and len({x['name'] for x in c})==52
    assert {x['name'] for x in c if 'grid' in x}=={f'AIZ-excite-{d:g}@{g}' for d in (.003,.01) for g in ('neural-half','body-half')}
    assert min(x['conductance_nS'] for x in c)==.0003 and all(retains_trace('e097',x) for x in c)
    assert len(cases())==60 and len(cases('e096'))==60 and not any('grid' in x for x in cases())
    assert retains_trace('e096',dict(conductance_nS=.1)) and not retains_trace('e096',dict(conductance_nS=.03))
    with pytest.raises(ValueError):cases('e098')
    probe=membrane_probe(['AIYL','AIYR','X'],next(x for x in c if x['group']=='AIY'),(40.,50.))
    assert np.allclose(probe(45.).conductance_nS[:2],.0003) and not probe(45.).conductance_nS[2]


def test_predict_at_interpolates_extrapolates_below_range_and_saturates_above():
    pts=[(.01,2.),(.1,20.),(1.,40.)]
    assert predict_at(pts,.1)==(pytest.approx(20.),'interpolated')
    assert predict_at(pts,.005)==(pytest.approx(1.),'extrapolated-below-range')
    assert predict_at(pts,5.)==(40.,'above-range') and predict_at(pts,0.)==(0.,'zero')
    assert predict_at([(0.,3.)],.1)==(0.,'zero')


def test_grid_controls_never_enter_the_minimum_dose_summary():
    table=[dict(status='completed',group='AIZ',sign='excite',dose_nS=.003,target_reached=True,grid='neural-half'),
        dict(status='completed',group='AIZ',sign='excite',dose_nS=.01,target_reached=True)]
    assert minimum_doses(table)=={'AIZ-excite':.01}


def test_calibration_pipeline_passes_the_e097_profile(tmp_path,monkeypatch):
    import scripts.total_job_budget as budget
    out=tmp_path/'out';calls=[]
    def run(cmd,seconds):
        calls.append(cmd)
        if 'scripts.audit_node_probe' in cmd:
            audit=tmp_path/'out-audit';audit.mkdir();(audit/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
        else:(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,all_stages_finished=True,validation_complete=True,sensory_motion_component_qualified=False)))
        return 0,False
    monkeypatch.setattr(budget,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','node-probe-calibration','--total-minutes','60','--output-dir',str(out),'--source-run',str(tmp_path/'source')])
    assert budget.main()==0 and calls[0][calls[0].index('--profile')+1]=='e097' and 'scripts.run_node_probe' in calls[0] and 'scripts.audit_node_probe' in calls[1]
    assert json.loads((out/'deadline.json').read_text())['experiment']=='E097'


def test_e097_coordinator_runs_baseline_cue_run_and_all_probes(tmp_path,monkeypatch):
    import scripts.run_node_probe as runner
    import scripts.benchmark_cpu as topology
    out=tmp_path/'run';out.mkdir();(out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+20000)))
    inputs=out/'inputs';inputs.mkdir()
    monkeypatch.setattr(runner,'prepare',lambda source,o,experiment,bilateral:(inputs,None,None,0,'test-parent','audit-path') if experiment=='E097' else 1/0)
    monkeypatch.setattr(runner,'capture_run_record',lambda *args:None)
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(128)),raising=False)
    monkeypatch.setattr(topology,'physical_core_cpus',lambda allowed:(list(range(128)),True))
    requests=[]
    def fake(j):
        requests.append(j);folder=Path(j['output']);folder.mkdir(parents=True)
        probe=j['probe'];hit=probe and probe['group']=='AIZ' and probe['reversal_mV']==0 and probe['conductance_nS']>=.01
        feature=data(effect=.002)['cue'] if j.get('cue') else data()['no_cue'] if probe is None else data(effect=.1)['cue'] if hit else data(effect=0.)['cue']
        return dict(output=j['output'],status='completed',features=feature,checkpoint_sha256='test-parent',trajectory_sha256={'trace':'x'},probe=probe,cue=bool(j.get('cue')))
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,j):
            f=concurrent.futures.Future();f.set_result(fake(j));return f
    monkeypatch.setattr(runner.concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(sys,'argv',['probe','--output-dir',str(out),'--source-run',str(tmp_path/'source'),'--profile','e097'])
    assert runner.main()==0
    s=json.loads((out/'summary.json').read_text());config=json.loads((out/'configuration.json').read_text())
    assert config['profile']=='e097' and s['experiment']=='E097' and len(requests)==54 and s['evaluation_count']==54 and s['all_stages_finished']
    assert [j.get('cue',False) for j in requests].count(True)==1 and all(j['retain_traces'] for j in requests)
    assert len(s['probe_table'])==52 and s['cue_reduction_percent'][0]==pytest.approx(.2)
    assert s['minimum_doses_reaching_target_nS']=={'AIZ-excite':.01}
    assert not s['sensory_motion_component_qualified']


def test_e097_audit_checks_cue_run_and_skips_ordinary_grid_comparisons_for_grid_controls(tmp_path):
    from scripts.audit_node_probe import audit
    from scripts.experiment_records import file_sha256
    _,traces,ids=paired()
    run=tmp_path/'run';inputs=run/'inputs';inputs.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    np.savez(inputs/'motion_parent.npz',neuron_ids=np.array(ids),fixed_release_threshold_mV=np.full(len(ids),-40.),fixed_release_slope_mV=np.full(len(ids),5.))
    reference=tmp_path/'external-trace';reference.write_bytes(b'reference evidence')
    write(inputs/'sensory_reference.json',dict(path=str(reference),sha256=file_sha256(reference)))
    write(inputs/'manifest.json',dict(sha256={p.name:file_sha256(p) for p in inputs.iterdir()}))
    planned=cases('e097');write(run/'configuration.json',dict(profile='e097',parent_sha256=file_sha256(inputs/'motion_parent.npz'),planned=planned,probe_window_s=[40.,50.]))
    def feat(trace):return features(trace,ids,'nominal',-40.,5.)
    def evaluation(label,probe,trace,cue=False):
        folder=run/'evaluations'/label;folder.mkdir(parents=True)
        write(folder/'request.json',dict(output=str(folder),probe=probe,retain_traces=True,until=1.,**(dict(cue=True) if cue else {})))
        np.savez_compressed(folder/'trace.npz',**trace)
        r=dict(status='completed',features=feat(trace),checkpoint_sha256=file_sha256(inputs/'motion_parent.npz'),trajectory_sha256=dict(trace=file_sha256(folder/'trace.npz')),
            output=str(folder),elapsed_seconds=1.,probe=probe,cue=cue)
        write(folder/'summary.json',r);return r
    def row(case,base,r):
        m=score(dict(no_cue=base['features'],cue=r['features']),base['features'],TARGET,1.)
        out=dict(output=r['output'],name=case['name'],group=case['group'],sign='excite',dose_nS=case['conductance_nS'],status='completed',
            reduction_percent=reduction_percent(m),regional_motion_ratios=m['regional_motion_ratios'][0],safe=m['safe'],
            baseline_and_recovery_preserved=m['baseline_and_recovery_preserved'],regional_cycle_ratios=m['regional_cycle_ratios'][0],target_reached=reaches_target(m))
        if 'grid' in case:out['grid']=case['grid']
        return out
    base=evaluation('baseline',None,traces['no_cue']);cue=evaluation('cue-run',None,traces['cue'],cue=True)
    grid=next(c for c in planned if c['name']=='AIZ-excite-0.003@neural-half')
    probe=evaluation(grid['name'],grid,traces['cue'])
    table=[row(grid,base,probe)];reduction=reduction_percent(score(dict(no_cue=base['features'],cue=cue['features']),base['features'],TARGET,1.))
    summary=dict(experiment='E097',profile='e097',evaluation_count=3,baseline=base,cue_run=cue,cue_reduction_percent=reduction,probes=[probe],probe_table=table,
        minimum_doses_reaching_target_nS=minimum_doses(table),validation_complete=False)
    write(run/'summary.json',summary)
    assert audit(run,tmp_path/'audit')==0
    summary['cue_reduction_percent']=[x+1. for x in reduction];write(run/'summary.json',summary)
    assert audit(run,tmp_path/'bad-audit')==2
    assert any('Cue-run reduction mismatch' in p for p in json.loads((tmp_path/'bad-audit/summary.json').read_text())['problems'])


def test_calibration_analysis_compares_cue_node_change_with_the_probe_curve(tmp_path,monkeypatch):
    import scripts.analyze_node_calibration as analysis
    from scripts.experiment_records import file_sha256
    ids=['RIAL','RIAR','AIYL'];t=np.arange(0,70.001,.1)
    run=tmp_path/'node-calibration-run';(run/'inputs').mkdir(parents=True);audit=tmp_path/'node-calibration-run-audit';audit.mkdir()
    np.savez(run/'inputs/motion_parent.npz',neuron_ids=np.array(ids),fixed_release_threshold_mV=np.full(3,-40.),fixed_release_slope_mV=np.full(3,5.))
    def trace(label,shift):
        folder=run/'evaluations'/label;folder.mkdir(parents=True);v=np.full((len(t),3),-45.);v[(t>=41)&(t<46),:2]+=shift
        np.savez_compressed(folder/'trace.npz',dense_time_s=t,dense_voltage_mV=v)
        return dict(output=str(folder),status='completed',trajectory_sha256=dict(trace=file_sha256(folder/'trace.npz')))
    base=trace('baseline',0.);cue=trace('cue-run',.2);probes=[];table=[]
    for dose,shift,red in ((.001,.1,[1.,2.,2.]),(.003,.5,[3.,10.,10.]),(.01,2.,[9.,20.,20.])):
        r=trace(f'RIA-excite-{dose:g}',shift);probes.append(r)
        table.append(dict(output=r['output'],name=f'RIA-excite-{dose:g}',group='RIA',sign='excite',dose_nS=dose,status='completed',reduction_percent=red,baseline_and_recovery_preserved=True))
    (run/'summary.json').write_text(json.dumps(dict(experiment='E097',validation_complete=True,baseline=base,cue_run=cue,cue_reduction_percent=[.05,.1,.1],probes=probes,probe_table=table)))
    (run/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True)));(audit/'summary.json').write_text(json.dumps(dict(integrity_passed=True)))
    monkeypatch.setattr(analysis,'CALIBRATION_GROUPS',('RIA',))
    g=analysis.analyze(run,tmp_path/'calibration')['groups']['RIA']
    assert g['matching_probe_sign']=='excite' and g['cue_node_release_change']>0 and g['prediction_status']=='interpolated'
    assert .001<g['probe_equivalent_dose_nS']<.003 and 1.<g['predicted_reduction_percent_at_cue_node_change']['tail']<3.
    assert set(g['cue_per_neuron_release_change'])=={'RIAL','RIAR'} and (tmp_path/'calibration/summary.json').exists()

