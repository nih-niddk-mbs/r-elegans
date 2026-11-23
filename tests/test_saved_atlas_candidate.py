"""Synthetic fixed-candidate assessment/audit tests; no scientific GPU run."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.experiment_records import file_sha256
from scripts.stable_atlas_worker import atomic_json as write
from scripts.selected_atlas_audit_checks import saved_arrays


@pytest.mark.parametrize('improve',[False,True])
def test_single_candidate_full_validation_audit_and_source_unchanged(tmp_path,monkeypatch,improve):
    import scripts.run_saved_atlas_candidate as run
    from scripts.stable_atlas_policy import Coordinates,loss
    from r_elegans.brain.rhythm_qualification import rhythm_metrics,compare_cycles
    from test_atlas_target_refinement import BUNDLE
    root=tmp_path/'run';root.mkdir();inputs=root/'inputs';inputs.mkdir();source=tmp_path/'source';source.mkdir()
    parent=inputs/'parent.npz';parent.write_bytes(BUNDLE.read_bytes());origin=saved_arrays(parent);codec=Coordinates(origin)
    x=np.full(codec.size,-.000025);candidate=inputs/'candidate.npz';np.savez_compressed(candidate,**codec.apply(x))
    coords=inputs/'coords.npz';np.savez_compressed(coords,offsets=x)
    absfile=inputs/'absolute.npz';np.savez_compressed(absfile,offsets=np.zeros_like(x))
    ids=origin['neuron_ids'].tolist();targets=ids[:3];pairs=[[t,r] for t in targets for r in ids[20:27]]
    stats=dict(stats={s:dict(pairs=pairs,mean_z=np.linspace(-1,1,len(pairs)).tolist()) for s in ('train','val','test')})
    st=inputs/'statistics.json';write(st,stats)
    base=np.random.default_rng(8).normal(0,1e-5,(4,3,302));response=base.copy()
    if improve:
        for pair,z in zip(pairs,stats['stats']['train']['mean_z']):response[:,targets.index(pair[0]),ids.index(pair[1])]=z*1e-5
    baselines=[]
    for level in range(3):
        path=inputs/f'base-{level}.npz';np.savez_compressed(path,response=base);baselines.append(str(path))
    v=20*np.sin(2*np.pi*2*np.arange(12000)*.001)-40;trace=np.tile(v[None,:,None],(4,1,2));rm=[[rhythm_metrics(v,.001)]*2 for _ in range(4)]
    vv=20*np.sin(2*np.pi*2*np.arange(20000)*.002)-40;longrm=[[rhythm_metrics(vv,.002)]*2 for _ in range(2)]
    strict=[compare_cycles(vv,vv,.002)]*2
    write(source/'autonomy.json',dict(selected=dict(rhythms=longrm)))
    steps=(.125,.0625,.03125,.0078125,.00390625,.001953125)
    m=dict(experiment='E120',parent=str(parent),parent_sha256=file_sha256(parent),candidate=str(candidate),candidate_sha256=file_sha256(candidate),
        coordinates=str(coords),parent_absolute_offsets=str(absfile),statistics=str(st),targets=targets,panel=targets,
        hard=[targets[0]],baselines=baselines,stimulus_nS=.03,rhythm_references={str(step):rm for step in steps},
        panel_cached_loss=loss(response,targets,ids,stats,'train'),source_parent_run=str(source),
        evidence_sha256={str(parent):file_sha256(parent),str(candidate):file_sha256(candidate),str(coords):file_sha256(coords)})
    before={p:file_sha256(p) for p in (parent,candidate,coords)}
    def prepare(src,out):write(out/'manifest.json',m);return m
    monkeypatch.setattr(run,'prepare',prepare)
    write(root/'deadline.json',dict(source='mocked',coordinator_deadline=100000000000))
    class FakePool:
        def __init__(self,root):pass
        def close(self):pass
        def tasks(self,tasks,deadline):
            rows=[]
            for t in tasks:
                folder=Path(t['result']).parent;folder.mkdir(parents=True);write(folder/'task.json',t);out=Path(t['output']);out.mkdir()
                if t['kind'] in ('anchors','autonomy'):
                    anchor=out/'anchors.npz';np.savez_compressed(anchor,model_sha256=t['model_sha256'],**{'start_'+k:np.zeros((4,302,35)) if k=='gates' else np.zeros((4,302)) for k in ('voltage_mV','gates','calcium_uM','indicator')})
                    r=dict(anchors=str(anchor),anchors_sha256=file_sha256(anchor))
                    if t['kind']=='autonomy':
                        for step in (.03125,.015625):np.savez_compressed(out/f'rhythm-{step}.npz',rmd_mV=np.tile(vv[:,None],(1,2)))
                        r.update(rhythms=longrm,strict_cycle_checks=strict)
                else:
                    names=t['targets'];n=len(names);count=2*((n+7)//8)+(-n)%8;path=out/'responses.npz'
                    labels=[f'batch-{begin//8}-condition-{k}' for begin in range(0,n,8) for k in range(min(8,n-begin),10)]
                    np.savez_compressed(path,response=response[:,[targets.index(name) for name in names]],targets=names,neuron_ids=ids,
                        control_means=np.zeros((count,4,302)),reference_sham_rmd_mV=trace,control_labels=labels,
                        maximum_control_voltage_delta_mV=0.,neural_step_ms=t['step_ms'],anchors_sha256=t['anchors_sha256'])
                    r=dict(artifact=str(path),sha256=file_sha256(path),controls_passed=True,duplicate_mean_error=0.,duplicate_trace_error=0.,rhythms=rm)
                r.update(task=t,status='completed',elapsed_seconds=1.);write(t['result'],r);rows.append(r)
            return rows
    monkeypatch.setattr(run,'Pool',FakePool)
    run.coordinator(root);s=json.loads((root/'summary.json').read_text())
    assert s['candidate_qualified_before_audit']==improve
    assert s['gates']['panel_reproduced'] and not s['model_promoted'] and s['evaluation_only']
    assert before=={p:file_sha256(p) for p in before}
    a=run.audit(root);assert a['integrity_passed'],a['problems'];assert a['execution_complete']
    audit_sha=file_sha256(root/'audit.json')
    independent=run.audit(root,output_dir=tmp_path/'independent-audit')
    assert independent['integrity_passed'] and independent['execution_complete']
    assert file_sha256(root/'audit.json')==audit_sha
    data=saved_arrays(candidate);data['fixed_bias_pA'][0]+=1;np.savez_compressed(candidate,**data)
    assert not run.audit(root)['integrity_passed']


def test_incomplete_candidate_audit(tmp_path):
    from scripts.run_saved_atlas_candidate import audit
    result=audit(tmp_path)
    assert result['audit_complete'] and not result['execution_complete'] and result['unavailable_evidence']


def test_source_requires_clean_completed_direction_check(tmp_path):
    from scripts.run_saved_atlas_candidate import prepare
    source=tmp_path/'source';source.mkdir();out=tmp_path/'out';out.mkdir()
    write(source/'manifest.json',dict(experiment='E119'))
    write(source/'summary.json',{})
    write(source/'audit.json',dict(audit_complete=True,execution_complete=True,integrity_passed=True,problems=[],unavailable_evidence=[]))
    write(source/'budget_receipt.json',dict(all_steps_finished=True,diagnostic_checks_passed=False,timed_out=False))
    with pytest.raises(ValueError,match='E119'):prepare(source,out)
