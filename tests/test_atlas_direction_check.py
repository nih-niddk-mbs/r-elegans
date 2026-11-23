"""E119 deterministic curvature/readonly/audit tests; no GPU scientific run."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.experiment_records import file_sha256
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.stable_atlas_worker import atomic_json as write


def test_central_slopes_and_bound_refusal():
    from scripts.run_atlas_direction_check import curve,scaled_offsets,CASES
    rows={name:dict(losses={'1':1+2*a+3*a*a,'2':1+2*a+3*a*a}) for name,a in CASES.items()}
    result=curve(rows,[1,2])
    for row in result['2']:assert row['central_slope']==pytest.approx(2)
    np.testing.assert_array_equal(scaled_offsets(np.array([.001]),.1,np.array([.002])),[.0001])
    with pytest.raises(ValueError):scaled_offsets(np.array([.1]),.1,np.array([.099]))


def test_complete_direction_diagnostic_and_audit(tmp_path,monkeypatch):
    import scripts.run_atlas_direction_check as run
    from scripts.stable_atlas_policy import Coordinates,loss
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    from test_atlas_target_refinement import BUNDLE
    root=tmp_path/'run';root.mkdir();inputs=root/'inputs';inputs.mkdir()
    parent=inputs/'parent.npz';parent.write_bytes(BUNDLE.read_bytes());digest=file_sha256(parent)
    origin=saved_arrays(parent);codec=Coordinates(origin);ids=origin['neuron_ids'].tolist();names=ids[:3]
    pairs=[[t,r] for t in names for r in ids[20:27]];stats=dict(stats={s:dict(pairs=pairs,mean_z=np.linspace(-1,1,len(pairs)).tolist()) for s in ('train','val','test')})
    stat=inputs/'statistics.json';write(stat,stats)
    base=np.random.default_rng(6).normal(0,1e-5,(4,3,302));effect=np.random.default_rng(7).normal(0,1e-6,base.shape)
    optimizer=inputs/'optimizer.npz';np.savez_compressed(optimizer,proposal=np.full(codec.size,.00025),averaged_gradient=np.ones(codec.size))
    absolute=inputs/'absolute.npz';np.savez_compressed(absolute,offsets=np.zeros(codec.size))
    v=20*np.sin(2*np.pi*2*np.arange(12000)*.001)-40;trace=np.tile(v[None,:,None],(4,1,2));rhythm=[[rhythm_metrics(v,.001)]*2 for _ in range(4)]
    steps=(.125,.0625,.03125,.0078125,.00390625,.001953125)
    m=dict(experiment='E119',parent=str(parent),parent_sha256=digest,statistics=str(stat),optimizer=str(optimizer),
        parent_absolute_offsets=str(absolute),panel=names,hard=[names[0]],stimulus_nS=.03,evidence_sha256={str(parent):digest},
        cached_endpoints=dict(parent=loss(base,names,ids,stats,'train'),proposal=None),rhythm_references={str(step):rhythm for step in steps})
    def prepare(source,out):write(out/'manifest.json',m);return m
    monkeypatch.setattr(run,'prepare',prepare)
    write(root/'deadline.json',dict(source='mocked',coordinator_deadline=100000000000))
    class FakePool:
        def __init__(self,root):pass
        def close(self):pass
        def tasks(self,tasks,deadline):
            rows=[]
            for t in tasks:
                folder=Path(t['result']).parent;folder.mkdir(parents=True);write(folder/'task.json',t);out=Path(t['output']);out.mkdir()
                if t['kind']=='anchors':
                    anchor=out/'anchors.npz';np.savez_compressed(anchor,model_sha256=t['model_sha256'],**{'start_'+k:np.zeros((4,302,35)) if k=='gates' else np.zeros((4,302)) for k in run.FIELDS})
                    r=dict(anchors=str(anchor),anchors_sha256=file_sha256(anchor))
                else:
                    alpha=run.CASES[Path(t['model']).parent.name];targets=t['targets'];n=len(targets)
                    response=(base+alpha*effect)[:,[names.index(name) for name in targets]]
                    count=2*((n+7)//8)+(-n)%8;artifact=out/'responses.npz'
                    labels=[f'batch-{begin//8}-condition-{k}' for begin in range(0,n,8) for k in range(min(8,n-begin),10)]
                    np.savez_compressed(artifact,response=response,targets=targets,neuron_ids=ids,control_means=np.zeros((count,4,302)),
                        reference_sham_rmd_mV=trace,maximum_control_voltage_delta_mV=0.,control_labels=labels,
                        neural_step_ms=t['step_ms'],anchors_sha256=t['anchors_sha256'])
                    r=dict(artifact=str(artifact),sha256=file_sha256(artifact),controls_passed=True,duplicate_mean_error=0.,duplicate_trace_error=0.,rhythms=rhythm)
                r.update(task=t,status='completed',elapsed_seconds=1.);write(t['result'],r);rows.append(r)
            return rows
    monkeypatch.setattr(run,'Pool',FakePool)
    run.coordinator(root);s=json.loads((root/'summary.json').read_text())
    assert s['parent_reproduced'] and s['repeat_passed'] and s['all_panel_screens_passed']
    assert not s['fit_qualified'] and not s['model_promoted'] and file_sha256(parent)==digest
    a=run.audit(root);assert a['integrity_passed'],a['problems'];assert a['execution_complete']
    path=root/'trials/plus-small/network.npz';arrays=saved_arrays(path);arrays['fixed_bias_pA'][0]+=1;np.savez_compressed(path,**arrays)
    assert not run.audit(root)['integrity_passed']


def test_incomplete_direction_check_audit(tmp_path):
    from scripts.run_atlas_direction_check import audit
    result=audit(tmp_path)
    assert result['audit_complete'] and not result['execution_complete'] and result['unavailable_evidence']
