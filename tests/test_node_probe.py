import concurrent.futures,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from r_elegans.node_probe import cases,membrane_probe,GROUPS,DOSES,TARGET,reduction_percent,reaches_target,minimum_doses
from r_elegans.sensory_motion import features,score
from test_sensory_motion import data
from test_rial_sensory import paired


def test_cases_cover_ten_groups_both_signs_and_three_doses_without_duplicates():
    c=cases();assert len(c)==len(GROUPS)*2*len(DOSES)==60 and len({x['name'] for x in c})==60
    assert {x['reversal_mV'] for x in c}=={0.,-80.} and max(x['conductance_nS'] for x in c)<=.1


def test_probe_is_confined_to_named_neurons_and_the_cue_window_and_rejects_outside_envelope():
    ids=['A','RIML','RIMR','B'];case=dict(group='RIM',conductance_nS=.03,reversal_mV=-80.)
    probe=membrane_probe(ids,case,(40.,50.))
    inside=probe(45.);outside=probe(51.);before=probe(39.)
    assert np.allclose(inside.conductance_nS,[0,.03,.03,0]) and np.allclose(inside.weighted_reversal_pA,[0,-2.4,-2.4,0])
    assert not np.any(outside.conductance_nS) and not np.any(before.conductance_nS)
    for bad in (dict(case,conductance_nS=.2),dict(case,conductance_nS=0.),dict(case,reversal_mV=-40.),dict(case,group='AWA')):
        with pytest.raises(ValueError):membrane_probe(ids,bad,(40.,50.))


def test_minimum_dose_summary_uses_only_completed_cases_that_reach_the_gate():
    table=[dict(status='completed',group='RIM',sign='excite',dose_nS=.1,target_reached=True),
        dict(status='completed',group='RIM',sign='excite',dose_nS=.03,target_reached=True),
        dict(status='completed',group='RIM',sign='inhibit',dose_nS=.01,target_reached=False),dict(status='time_budget',group='RIB',sign='excite',dose_nS=.01)]
    assert minimum_doses(table)=={'RIM-excite':.03}


def test_coordinator_runs_baseline_and_every_probe_and_summarizes_the_dose_response(tmp_path,monkeypatch):
    import scripts.run_node_probe as runner
    import scripts.benchmark_cpu as topology
    out=tmp_path/'run';out.mkdir();(out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+20000)))
    inputs=out/'inputs';inputs.mkdir()
    monkeypatch.setattr(runner,'prepare',lambda source,o,experiment,bilateral:(inputs,None,None,0,'test-parent','audit-path'))
    monkeypatch.setattr(runner,'capture_run_record',lambda *args:None)
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(128)),raising=False)
    monkeypatch.setattr(topology,'physical_core_cpus',lambda allowed:(list(range(128)),True))
    pair=data(effect=.12);requests=[]
    def fake(j):
        requests.append(j);folder=Path(j['output']);folder.mkdir(parents=True)
        probe=j['probe'];hit=probe and probe['group']=='RIM' and probe['reversal_mV']==0 and probe['conductance_nS']>=.1
        feature=pair['no_cue'] if probe is None else (pair['cue'] if hit else data(effect=0.)['cue'])
        return dict(output=j['output'],status='completed',features=feature,checkpoint_sha256='test-parent',trajectory_sha256={},probe=probe)
    class Pool:
        def __init__(self,**kwargs):self.workers=kwargs['max_workers']
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,func,j):
            f=concurrent.futures.Future();f.set_result(fake(j));return f
    monkeypatch.setattr(runner.concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(sys,'argv',['probe','--output-dir',str(out),'--source-run',str(tmp_path/'source')])
    assert runner.main()==0
    s=json.loads((out/'summary.json').read_text());config=json.loads((out/'configuration.json').read_text())
    assert config['workers']==21 and len(config['cpu_groups'])==21
    assert len(requests)==61 and s['evaluation_count']==61 and s['all_stages_finished'] and len(s['probe_table'])==60
    assert s['minimum_doses_reaching_target_nS']=={'RIM-excite':.1}
    assert {j['probe']['name'] for j in requests if j['probe'] and j['retain_traces']}=={f'{g}-{l}-0.1' for g in GROUPS for l in ('excite','inhibit')}
    assert not s['sensory_motion_component_qualified'] and not s['full_start_stop_qualified']


def test_node_probe_budget_dispatches_runner_and_its_own_audit(tmp_path,monkeypatch):
    import scripts.total_job_budget as budget
    out=tmp_path/'out';calls=[]
    def run(cmd,seconds):
        calls.append(cmd)
        if 'scripts.audit_node_probe' in cmd:
            audit=tmp_path/'out-audit';audit.mkdir();(audit/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
        else:(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,all_stages_finished=True,validation_complete=True,sensory_motion_component_qualified=False)))
        return 0,False
    monkeypatch.setattr(budget,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','node-probe','--total-minutes','30','--output-dir',str(out),'--source-run',str(tmp_path/'source')])
    assert budget.main()==0 and 'scripts.run_node_probe' in calls[0] and '--objective' not in calls[0] and 'scripts.audit_node_probe' in calls[1]
    assert json.loads((out/'deadline.json').read_text())['experiment']=='E096'


def test_audit_recomputes_scores_checks_probe_sign_and_flags_a_mislabelled_probe(tmp_path):
    from scripts.audit_node_probe import audit
    from scripts.experiment_records import file_sha256
    _,traces,ids=paired()
    parent_npz=dict(neuron_ids=np.array(ids),fixed_release_threshold_mV=np.full(len(ids),-40.),fixed_release_slope_mV=np.full(len(ids),5.))
    run=tmp_path/'run';inputs=run/'inputs';inputs.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    np.savez(inputs/'motion_parent.npz',**parent_npz)
    reference=tmp_path/'external-trace';reference.write_bytes(b'reference evidence')
    write(inputs/'sensory_reference.json',dict(path=str(reference),sha256=file_sha256(reference)))
    write(inputs/'manifest.json',dict(sha256={p.name:file_sha256(p) for p in inputs.iterdir()}))
    planned=cases();write(run/'configuration.json',dict(parent_sha256=file_sha256(inputs/'motion_parent.npz'),planned=planned,probe_window_s=[40.,50.]))
    base_trace=traces['no_cue'];probe_trace=traces['cue'].copy()
    voltage=probe_trace['dense_voltage_mV'].copy();t=probe_trace['dense_time_s']
    on=(t>=40)&(t<50)
    for name in ('RIAL','RIAR'):voltage[on,ids.index(name)]+=1.
    probe_trace['dense_voltage_mV']=voltage
    def feat(trace):return features(trace,ids,'nominal',-40.,5.)
    def evaluation(label,probe,trace,retain=True):
        folder=run/'evaluations'/label;folder.mkdir(parents=True)
        write(folder/'request.json',dict(output=str(folder),probe=probe,retain_traces=retain,until=1.))
        np.savez_compressed(folder/'trace.npz',**trace)
        r=dict(status='completed',features=feat(trace),checkpoint_sha256=file_sha256(inputs/'motion_parent.npz'),
            trajectory_sha256=dict(trace=file_sha256(folder/'trace.npz')),output=str(folder),elapsed_seconds=1.,probe=probe)
        write(folder/'summary.json',r);return r
    def row(case,base,r):
        m=score(dict(no_cue=base['features'],cue=r['features']),base['features'],TARGET,1.)
        return dict(output=r['output'],name=case['name'],group=case['group'],sign='excite' if case['reversal_mV']==0 else 'inhibit',dose_nS=case['conductance_nS'],status='completed',
            reduction_percent=reduction_percent(m),regional_motion_ratios=m['regional_motion_ratios'][0],safe=m['safe'],
            baseline_and_recovery_preserved=m['baseline_and_recovery_preserved'],regional_cycle_ratios=m['regional_cycle_ratios'][0],target_reached=reaches_target(m))
    def summarize(base,done):
        table=[row(c,base,r) for c,r in done]
        write(run/'summary.json',dict(experiment='E096',evaluation_count=1+len(done),baseline=base,probes=[r for _,r in done],probe_table=table,
            minimum_doses_reaching_target_nS=minimum_doses(table),validation_complete=False))
    base=evaluation('baseline',None,base_trace)
    excite=next(c for c in planned if c['name']=='RIA-excite-0.1');inhibit=next(c for c in planned if c['name']=='RIA-inhibit-0.1')
    first=evaluation(excite['name'],excite,probe_trace);summarize(base,[(excite,first)])
    assert audit(run,tmp_path/'audit')==0
    second=evaluation(inhibit['name'],inhibit,probe_trace);summarize(base,[(excite,first),(inhibit,second)])
    assert audit(run,tmp_path/'bad-audit')==2
    problems=json.loads((tmp_path/'bad-audit/summary.json').read_text())['problems']
    assert any('wrong sign: RIA-inhibit-0.1' in p for p in problems) and not any('RIA-excite-0.1' in p for p in problems)
