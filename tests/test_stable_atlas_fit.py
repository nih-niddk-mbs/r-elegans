"""E113 software checks; synthetic signals, no scientific Mac fitting run."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.stable_atlas_policy import Coordinates, loss, prior, adam_step, choose_batch, preserved, admit
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.stable_atlas_worker import atomic_json as write
from scripts.experiment_records import file_sha256

BUNDLE = Path(__file__).parents[1]/'r_elegans/data/builtin/rhythm_network.npz'


def network():
    from test_atlas_target_refinement import BUNDLE as bundle
    return saved_arrays(bundle)


def test_mask_bounds_reciprocal_gap_and_frozen_fields():
    origin = network();codec = Coordinates(origin)
    rng = np.random.default_rng(12);x = rng.uniform(-.1,.1,codec.size);candidate = codec.apply(x)
    assert codec.audit(candidate,x)['passed']
    ids = origin['neuron_ids'].tolist();frozen = [i for i,n in enumerate(ids) if n.startswith('RMD')]
    np.testing.assert_array_equal(candidate['params_log_channel_scale'][frozen], origin['params_log_channel_scale'][frozen])
    np.testing.assert_array_equal(candidate['params_log_channel_scale'][:,16], origin['params_log_channel_scale'][:,16])
    delta = candidate['params_raw_gap']-origin['params_raw_gap']
    np.testing.assert_allclose(delta,delta.T,atol=1e-15,rtol=0)
    candidate['fixed_bias_pA'][0] += 1
    assert not codec.audit(candidate,x)['passed']
    with pytest.raises(ValueError):codec.apply(np.full(codec.size,.1001))
    with pytest.raises(ValueError):codec.apply(np.full(codec.size,np.nan))


def stats_fixture():
    ids = ['T0','T1','T2','T3','T4','T5','T6','T7']
    pairs = [[t,r] for t in ids[:3] for r in ids if r != t]
    split = dict(pairs=pairs,mean_z=np.linspace(-1,1,len(pairs)).tolist())
    return ids, dict(stats={key:dict(split) for key in ['train','val','test']})


def test_loss_matches_existing_implementation_and_ignores_self_test():
    import jax.numpy as jnp
    from scripts.train_atlas_gradient import correlation_loss, dense_pairs
    ids,stats = stats_fixture();targets = ids[:3];r = np.random.default_rng(4).normal(size=(3,8))
    z,w = dense_pairs(stats['stats']['train'],targets,ids,5)
    expected = float(correlation_loss(jnp.asarray(r),z,w)[0])
    assert loss(r,targets,ids,stats,'train') == pytest.approx(expected,abs=1e-12)
    before = loss(r,targets,ids,stats,'train')
    stats['stats']['test']['mean_z']=[-99]*len(stats['stats']['test']['pairs'])
    for i in range(3):r[i,i]=1e6
    assert loss(r,targets,ids,stats,'train') == before
    assert prior(np.zeros(3)) == 0


def test_adam_is_bounded_and_direction_points_downhill():
    x=np.zeros(10);proposed,m,v=adam_step(x,x,x,np.ones(10),1)
    assert (proposed<0).all() and (m>0).all() and (v>0).all()
    p,_,_=adam_step(np.full(10,-.1),m,v,np.ones(10),2)
    assert (p>=-.1).all()
    assert not admit(100,200,100) and admit(100,1000,100)


def test_batch_and_rhythm_rules():
    rng=np.random.default_rng(1130)
    assert 'fine' in choose_batch(rng,['fine','coarse','other'],['fine'],5)
    with pytest.raises(ValueError):choose_batch(rng,['one'],[],0)
    ref=dict(passed=True,median_period_s=.5,window_amplitudes_mV=[20.]*5)
    assert preserved(ref,ref)
    assert not preserved(ref,dict(ref,median_period_s=.6))
    assert not preserved(ref,dict(ref,window_amplitudes_mV=[10.]*5))


def panel(tmp_path, factor=1., levels=(1,2)):
    from scripts.run_stable_atlas_fit import COARSE_STEPS
    ids,stats=stats_fixture();targets=ids[:3]
    rhythm=dict(passed=True,median_period_s=.5,window_amplitudes_mV=[20.]*5)
    manifest=dict(hard=[],rhythm_references={str(COARSE_STEPS[l]):[[rhythm]*2 for _ in range(4)] for l in levels})
    results=[]
    for l in levels:
        a=np.tile(np.linspace(-1e-4,1e-4,24).reshape(1,3,8),(4,1,1))
        if l==levels[-1]:a*=factor
        path=tmp_path/f'level-{l}.npz'
        np.savez_compressed(path,response=a,targets=targets,anchors_sha256='anchor')
        results.append(dict(task=dict(targets=targets,step_ms=COARSE_STEPS[l],anchors_sha256='anchor'),
            artifact=str(path),sha256=file_sha256(path),controls_passed=True,rhythms=[[rhythm]*2 for _ in range(4)]))
    return results,manifest,stats,ids,targets


def test_screen_and_final_refuse_numerical_failure_and_coverage(tmp_path):
    from scripts.run_stable_atlas_fit import analyze_results
    r,m,s,ids,t=panel(tmp_path)
    metrics,_=analyze_results(r,m,s,ids,t,[1,2],'train')
    assert metrics['passed'] and metrics['observed_pairs']==21
    r[-1]['controls_passed']=False
    assert not analyze_results(r,m,s,ids,t,[1,2],'train')[0]['passed']
    with pytest.raises(ValueError):analyze_results(r[:1],m,s,ids,t,[1,2],'train')
    r,m,s,ids,t=panel(tmp_path,factor=2)
    assert not analyze_results(r,m,s,ids,t,[1,2],'val',final=True)[0]['passed']


def test_interrupted_run_audit_explicitly_unavailable(tmp_path):
    from scripts.audit_stable_atlas_fit import audit
    result=audit(tmp_path)
    assert result['audit_complete'] and not result['integrity_passed'] and not result['execution_complete']
    assert result['problems'] and not result['behavioral_gates']['swimming']


@pytest.mark.parametrize('own_state,improve,validation_regression,measured', [(False,False,False,False),(True,False,False,False),(True,True,False,False),(True,True,True,False),(True,True,False,True)])
def test_coordinator_and_complete_audit_with_synthetic_gpu_pool(tmp_path,monkeypatch,own_state,improve,validation_regression,measured):
    """Drive one actual optimizer round/final selection and complete saved-data audit."""
    import types
    import scripts.run_stable_atlas_fit as runner
    from scripts.audit_stable_atlas_fit import audit
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    root=tmp_path/'run';root.mkdir();arrays=network();ids=arrays['neuron_ids'].tolist()
    targets=ids[:3];pairs=[[t,r] for t in targets for r in ids[10:17]]
    stats=dict(targets=targets,stats={s:dict(pairs=pairs,mean_z=np.linspace(-1,1,len(pairs)).tolist()) for s in ('train','val','test')})
    inputs=root/'inputs';inputs.mkdir();model=inputs/'parent.npz';np.savez_compressed(model,**arrays)
    anchors=inputs/'anchors.npz';np.savez_compressed(anchors,fake=np.zeros(4))
    statfile=inputs/'statistics.json';write(statfile,stats)
    base=np.random.default_rng(5).normal(0,1e-5,(4,3,302))
    baselines=[]
    for level in range(3):
        path=inputs/f'base-{level}.npz';np.savez_compressed(path,response=base);baselines.append(str(path))
    v=20*np.sin(2*np.pi*2*np.arange(12000)*.001)-40
    trace=np.tile(v[None,:,None],(4,1,2));rhythm=[[rhythm_metrics(v,.001)]*2 for _ in range(4)]
    m=dict(parent=str(model),parent_sha256=file_sha256(model),anchors=str(anchors),anchors_sha256=file_sha256(anchors),
        statistics=str(statfile),baselines=baselines,targets=targets,hard=[targets[0]],stimulus_nS=.03,
        rhythm_references={str(step):rhythm for step in runner.COARSE_STEPS+runner.FINE_STEPS},evidence_sha256={}, policy={})
    def prepare(source,folder):write(folder/'manifest.json',m);return m
    monkeypatch.setattr(runner,'prepare',prepare)
    rounds=iter([True,False]);monkeypatch.setattr(runner,'admit',lambda *a:next(rounds))
    clock=[100.];monkeypatch.setattr(runner,'time',types.SimpleNamespace(monotonic=lambda:clock[0]))
    write(root/'deadline.json',dict(started_monotonic=100.,deadline_monotonic=28900.,source='synthetic', own_state=own_state, measured_schedule=measured))
    class FakePool:
        def __init__(self,folder):self.root=folder
        def close(self):pass
        def tasks(self,tasks,deadline):
            results=[];clock[0]+=60
            for t in tasks:
                folder=Path(t['result']).parent;folder.mkdir(parents=True);write(folder/'task.json',t)
                out=Path(t['output']);out.mkdir()
                if t['kind'] in ('autonomy','anchors'):
                    anchor=out/'anchors.npz';np.savez_compressed(anchor,fake=np.zeros(4), model_sha256=t['model_sha256'])
                    vv=20*np.sin(2*np.pi*2*np.arange(20000)*.002)-40
                    rhythms=[[rhythm_metrics(vv,.002)]*2 for _ in range(2)]
                    for step in (.03125,.015625):np.savez_compressed(out/f'rhythm-{step}.npz',rmd_mV=np.tile(vv[:,None],(1,2)))
                    r=dict(anchors=str(anchor),anchors_sha256=file_sha256(anchor),rhythms=rhythms)
                else:
                    names=t['targets'];n=len(names);controls=2*((n+7)//8)+(-n)%8
                    a=base[:,[targets.index(name) for name in names]].copy()
                    changed = not np.array_equal(saved_arrays(t['model'])['params_log_channel_scale'], arrays['params_log_channel_scale'])
                    if improve and (t['name'].startswith('r1-plus-') or (changed and not t['name'].startswith('r1-minus-'))):
                        for pair, z in zip(pairs, stats['stats']['train']['mean_z']):
                            if pair[0] in names:
                                sign = -1 if validation_regression and t['name'].startswith('validation-') else 1
                                a[:,names.index(pair[0]),ids.index(pair[1])] = sign*z*1e-5
                    path=out/'responses.npz'
                    np.savez_compressed(path,response=a,targets=names,neuron_ids=ids,
                        anchors_sha256=t['anchors_sha256'],neural_step_ms=t['step_ms'],
                        control_means=np.zeros((controls,4,302)),maximum_control_voltage_delta_mV=0.,
                        reference_sham_rmd_mV=trace,control_labels=[f'batch-0-condition-{k}' for k in range(n,10)])
                    r=dict(artifact=str(path),sha256=file_sha256(path),controls_passed=True,rhythms=rhythm,
                        duplicate_mean_error=0.,duplicate_trace_error=0.)
                r.update(status='completed',task=t,elapsed_seconds=1.)
                write(t['result'],r);results.append(r)
            if measured and any(t['name'].startswith('validation-') for t in tasks):clock[0]=100.+330*60-1
            return results
    monkeypatch.setattr(runner,'Pool',FakePool)
    if measured:monkeypatch.setattr(runner,'admit',admit)
    runner.coordinator(root)
    summary=json.loads((root/'summary.json').read_text())
    assert summary['execution_complete'] and summary['validation_complete'] and summary['iterations']==1
    if own_state:assert summary['selection_reproduced']
    if improve:assert summary['accepted_updates'] == 1
    assert summary['fit_qualified_before_audit'] == (improve and not validation_regression)  # unchanged parent cannot masquerade as trained success
    result=audit(root)
    assert result['integrity_passed'],result['problems']
    assert result['execution_complete'] and result['candidate_parameter_checks'] and result['signal_and_control_checks']
    # Audit detects silent post-run changes to otherwise well-formed artifacts.
    chosen=Path(json.loads((root/'selection.json').read_text())['selected'])
    damaged=saved_arrays(chosen);damaged['fixed_bias_pA'][0]+=1
    np.savez_compressed(chosen,**damaged)
    assert not audit(root)['integrity_passed']


def test_measured_two_gpu_admission_uses_work_not_startup():
    from scripts.stable_atlas_policy import estimate_own_round, two_worker_seconds
    # E114 reported source replay and anchor seconds. Group counts below are a
    # synthetic full135-target projection; actual server group lengths read at runtime.
    targets=[f'C{i}' for i in range(123)]+[f'F{i}' for i in range(12)];hard=targets[123:]
    rows=[dict(status='completed',elapsed_seconds=2.41*60,task=dict(kind='anchors'))]
    for step,minutes,name in [(.0625,.36,'C0'),(.03125,.53,'C0'),(.00390625,3.92,'F0'),(.001953125,7.7,'F0')]:
        rows.append(dict(status='completed',elapsed_seconds=minutes*60,task=dict(kind='replay',step_ms=step,targets=[name])))
    estimate=estimate_own_round(rows,targets,hard,['C1','F1'])
    assert 50*60 < estimate['predicted_seconds'] < 70*60
    assert admit(26*60,120*60,estimate['predicted_seconds'])
    assert not admit(26*60,90*60,3*26*60)  # old startup multiplier skips all work
    assert not admit(100*60,120*60,estimate['predicted_seconds'])
    assert two_worker_seconds([10,1,1,10]) == 12
    with pytest.raises(ValueError):estimate_own_round([],targets,hard,['C1','F1'])
    with pytest.raises(ValueError):two_worker_seconds([float('nan')])
