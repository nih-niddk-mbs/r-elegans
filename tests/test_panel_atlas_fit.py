"""Synthetic E117 protocol checks; no claim of scientific GPU fitting."""
import json
import types
from pathlib import Path
import numpy as np
import pytest
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.stable_atlas_worker import atomic_json as write
from scripts.experiment_records import file_sha256


def test_train_only_stratification_and_no_silent_short_panel():
    from scripts.run_panel_atlas_fit import training_panel, expensive_first
    ids=[f'N{i}' for i in range(12)]
    train=dict(pairs=[[t,r] for t in ids[:9] for r in ids if t!=r])
    stats=dict(stats=dict(train=train,test=dict(pairs=[['TEST_ONLY','N0']])))
    p=training_panel(stats,['N0'],np.random.default_rng(1170))
    assert len(p)==8 and len(set(p))==8 and 'N0' in p and 'TEST_ONLY' not in p
    with pytest.raises(ValueError):training_panel(stats,[],np.random.default_rng(1170))
    tasks=[dict(step_ms=.1,targets=['a']),dict(step_ms=.01,targets=['a'])]
    assert expensive_first(tasks)[0]['step_ms']==.01


@pytest.mark.parametrize('improve',[False,True])
def test_full_proposal_selection_and_independent_audit(tmp_path,monkeypatch,improve):
    import scripts.run_panel_atlas_fit as pilot
    from scripts.stable_atlas_policy import Coordinates
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    from test_atlas_target_refinement import BUNDLE
    root=tmp_path/'run';root.mkdir();source=tmp_path/'source';source.mkdir()
    inputs=root/'inputs';inputs.mkdir();origin=saved_arrays(BUNDLE);ids=origin['neuron_ids'].tolist()
    parent=inputs/'parent.npz';np.savez_compressed(parent,**origin);codec=Coordinates(origin)
    names=ids[:9];pairs=[[t,r] for t in names for r in ids[20:27]]
    stats=dict(targets=names,stats={s:dict(pairs=pairs,mean_z=np.linspace(-1,1,len(pairs)).tolist()) for s in ('train','val','test')})
    st=inputs/'statistics.json';write(st,stats)
    base=np.random.default_rng(9).normal(0,1e-5,(4,9,302));baselines=[]
    for i in range(3):
        path=inputs/f'base-{i}.npz';np.savez_compressed(path,response=base);baselines.append(str(path))
    absfile=inputs/'absolute.npz';np.savez_compressed(absfile,offsets=np.zeros(codec.size))
    v=20*np.sin(2*np.pi*2*np.arange(12000)*.001)-40;trace=np.tile(v[None,:,None],(4,1,2))
    rm=[[rhythm_metrics(v,.001)]*2 for _ in range(4)]
    vv=20*np.sin(2*np.pi*2*np.arange(20000)*.002)-40;longrm=[[rhythm_metrics(vv,.002)]*2 for _ in range(2)]
    write(source/'autonomy.json',dict(selected=dict(rhythms=longrm)))
    steps=(.125,.0625,.03125,.0078125,.00390625,.001953125)
    manifest=dict(parent=str(parent),parent_sha256=file_sha256(parent),statistics=str(st),targets=names,hard=[names[0]],
        stimulus_nS=.03,baselines=baselines,parent_absolute_offsets=str(absfile),parent_absolute_offsets_sha256=file_sha256(absfile),
        evidence_sha256={str(source/'autonomy.json'):file_sha256(source/'autonomy.json')},
        rhythm_references={str(step):rm for step in steps},source=str(source),continuation={},policy={})
    def prepare(src,out):return manifest
    monkeypatch.setattr(pilot,'prepare_continuation',prepare)
    write(root/'deadline.json',dict(coordinator_deadline=100000000000,parent_run=str(source)))
    class FakePool:
        def __init__(self,root):pass
        def close(self):pass
        def tasks(self,tasks,deadline):
            result=[]
            for t in tasks:
                folder=Path(t['result']).parent;folder.mkdir(parents=True);write(folder/'task.json',t)
                out=Path(t['output']);out.mkdir()
                if t['kind'] in ('anchors','autonomy'):
                    anchor=out/'anchors.npz';np.savez_compressed(anchor,model_sha256=t['model_sha256'],
                        **{'start_'+k:np.zeros((4,302,35)) if k=='gates' else np.zeros((4,302)) for k in ('voltage_mV','gates','calcium_uM','indicator')})
                    r=dict(anchors=str(anchor),anchors_sha256=file_sha256(anchor))
                    if t['kind']=='autonomy':
                        for step in (.03125,.015625):np.savez_compressed(out/f'rhythm-{step}.npz',rmd_mV=np.tile(vv[:,None],(1,2)))
                        r['rhythms']=longrm
                else:
                    targets=t['targets'];n=len(targets);a=base[:,[names.index(name) for name in targets]].copy()
                    is_proposal=Path(t['model']).parent.name=='proposal'
                    better=improve and (('d0-plus' in t['model']) or ('d1-plus' in t['model']) or is_proposal)
                    if better:
                        for pair,z in zip(pairs,stats['stats']['train']['mean_z']):
                            if pair[0] in targets:a[:,targets.index(pair[0]),ids.index(pair[1])]=z*1e-5
                    count=2*((n+7)//8)+(-n)%8;path=out/'responses.npz'
                    labels=[f'batch-{begin//8}-condition-{k}' for begin in range(0,n,8) for k in range(min(8,n-begin),10)]
                    np.savez_compressed(path,response=a,targets=targets,neuron_ids=ids,control_means=np.zeros((count,4,302)),
                        reference_sham_rmd_mV=trace,maximum_control_voltage_delta_mV=0.,control_labels=labels,
                        anchors_sha256=t['anchors_sha256'],neural_step_ms=t['step_ms'])
                    r=dict(artifact=str(path),sha256=file_sha256(path),controls_passed=True,duplicate_mean_error=0.,duplicate_trace_error=0.,rhythms=rm)
                r.update(task=t,status='completed',elapsed_seconds=1.)
                write(t['result'],r);result.append(r)
            return result
    monkeypatch.setattr(pilot,'Pool',FakePool)
    pilot.coordinator(root);s=json.loads((root/'summary.json').read_text())
    assert s['fit_qualified_before_audit']==improve
    assert s['independent_full_loss_reproduction'] is False
    assert s['gates']['state_reproduced'] and s['gates']['panel_reproduced']
    result=pilot.audit(root);assert result['integrity_passed'],result['problems']
    assert result['execution_complete'] and not result['behavioral_qualification']
    path=root/'optimizer.npz';a=saved_arrays(path);a['averaged_gradient']+=1.;np.savez_compressed(path,**a)
    assert not pilot.audit(root)['integrity_passed']


def test_incomplete_audit_cannot_qualify(tmp_path):
    from scripts.run_panel_atlas_fit import audit, decision
    result=audit(tmp_path)
    assert result['audit_complete'] and not result['execution_complete'] and not result['integrity_passed']
    gates={k:True for k in ('parent_replay_passed','gradient_screen_passed','training_improved','validation_improved',
        'full_numerical_passed','parameters_passed','parameters_changed','autonomy_passed','state_reproduced','panel_reproduced')}
    assert decision(gates)
    for k in gates:assert not decision(dict(gates,**{k:False}))
