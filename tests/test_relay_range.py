from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import logit
from r_elegans.relay_range import features, score, parameter_groups
from scripts.audit_rial_sensory import relay_failures
from test_rial_sensory import paired


def relay_pair(baseline=.94, effect=.0001, recovery=True, blocked=False):
    _,traces,ids=paired(effect=0.);result={}
    for name,trace in traces.items():
        t=trace['dense_time_s'];r=np.full(len(t),baseline)
        if name=='cue' and not blocked:r[(t>=41)&(t<(46 if recovery else 70))]+=effect
        trace['dense_voltage_mV'][:,ids.index('AIYL')]=-40+5*logit(r)
        result[name]=features(trace,ids,'nominal',-40,5)
    return result


def test_range_alone_and_cue_alone_do_not_qualify():
    parent=relay_pair()['no_cue']
    good=relay_pair()
    assert score(good,parent,.02)['partial_attenuation_passed']
    assert not relay_failures(good,.02)
    for pair in (relay_pair(baseline=.99),relay_pair(effect=0.),relay_pair(recovery=False),relay_pair(blocked=True)):
        assert not score(pair,parent,.02)['partial_attenuation_passed']
        assert relay_failures(pair,.02)
    # Either direction is a relay-response component, never a motor-stop claim.
    negative=score(relay_pair(effect=-.0001),parent,.02)
    assert negative['partial_attenuation_passed'] and not negative['full_start_stop_qualified']
    weak=deepcopy(good)
    for condition in weak.values():
        for category in ('active','response','late'):
            for row in condition[category]:row['regional_cycles_rad']=[.001]*3
    assert not score(weak,parent,.02)['partial_attenuation_passed']


def test_range_objective_has_signal_before_gate_and_rejects_inconsistent_response():
    from r_elegans.relay_range import direction_consistent
    parent=relay_pair()['no_cue']
    losses=[score(relay_pair(baseline=x,effect=.000026),parent,.02,1.3)['loss'] for x in (.99,.985,.98,.97)]
    assert all(a>b for a,b in zip(losses[:3],losses[1:]))
    data=relay_pair();ref=data['no_cue']['response'][0]['AIYL_release_mean']
    data['cue']['response'][0]['AIYL_release_half_means']=[ref-.0001,ref+.0003]
    assert not score(data,parent,.02)['partial_attenuation_passed']
    assert direction_consistent([dict(metrics=dict(relay_response_mean_changes=[.001,.002]))])
    assert not direction_consistent([dict(metrics=dict(relay_response_mean_changes=[.001,-.002]))])


def test_focused_groups_freeze_other_channels_and_rial_motor_outputs():
    ids=('AWAL','AIYL','RIAL','SMDVL','AIYR');n=len(ids)
    channels=np.ones((n,17),bool);channels[:,16]=False
    chemical=np.ones((n,n),bool)
    spec=SimpleNamespace(channels=channels,chemical=chemical)
    size=channels.sum()+chemical.sum();origin=np.zeros(size)
    groups=parameter_groups(spec,ids,origin,-np.ones(size),np.ones(size))
    used=[i for g in groups for i in g['indices']]
    assert len(used)==len(set(used))
    ci,ei=np.nonzero(channels),np.nonzero(chemical);nc=len(ci[0])
    for i in used:
        if i<nc:assert ids[ci[0][i]] in ('AWAL','AIYL')
        else:
            post,pre=ei[0][i-nc],ei[1][i-nc]
            assert ids[post]=='AIYL' or (ids[pre],ids[post])==('AIYL','RIAL')


def test_relay_counterfactual_only_uses_existing_aiyl_rial_edge():
    from r_elegans.rial_sensory import replay_probe
    ids=('AWAL','AIYL','RIAL','SMDVL');mask=np.ones((4,4))
    probe=replay_probe(ids,mask,np.array([0.,70.]),np.array([.5,.6]),40.,.002,
                       source_name='AIYL',post_names=('RIAL',))
    np.testing.assert_array_equal(probe(45).pre_indices,[1])
    np.testing.assert_array_equal(probe(45).post_indices,[2])
    assert not bool(probe(30).active)


@pytest.mark.parametrize('variant',['nominal','redundant'])
def test_worker_relay_diagnostic_does_not_claim_body_mediation_without_body_effect(tmp_path,monkeypatch,variant):
    import time
    import scripts.rial_sensory_worker as worker
    import scripts.compare_rhythm_worker as verifier
    import r_elegans.brain.conductance_network as network
    _,traces,ids=paired(variant=variant,effect=0.);ai=ids.index('AIYL');ri=ids.index('RIAL');calls=[]
    for name,trace in traces.items():
        t=trace['dense_time_s'];release=np.full(len(t),.95)
        if name=='cue':
            release[(t>=41)&(t<46)]+=.0001
            if variant=='redundant':release[(t>=54)&(t<59)]+=.00001
        trace['dense_voltage_mV'][:,ai]=-40+5*logit(release)
    mask=np.zeros((len(ids),len(ids)));mask[ri,ai]=1
    fixed=SimpleNamespace(chemical_mask=mask,release_threshold_mV=np.full(len(ids),-40.),release_slope_mV=np.full(len(ids),5.))
    def factory(dt,brain,**kwargs):
        cue=bool(np.asarray(kwargs['world_override'].concentration_um)[:,0].max()>0)
        probe=kwargs.get('diagnostic_chemical_release');calls.append((cue,probe is not None))
        if probe is not None:
            np.testing.assert_array_equal(probe(45.).pre_indices,[ai])
            np.testing.assert_array_equal(probe(45.).post_indices,[ri])
        return lambda *args:traces['cue' if cue else 'no_cue']
    monkeypatch.setattr(worker,'_CACHE',{})
    monkeypatch.setattr(worker,'_DATA',dict(origin=np.zeros(1),lower=-np.ones(1),upper=np.ones(1),groups=[dict(indices=[0],lower=-.4,upper=.4)],
        decode=lambda x:x,fixed=fixed,config=None,model=SimpleNamespace(neuron_ids=ids),factory=factory,gains=np.ones(3),root=tmp_path))
    monkeypatch.setattr(verifier,'verify_inputs',lambda root:None)
    monkeypatch.setattr(network,'save_network',lambda path,*args:path.write_bytes(b'model'))
    monkeypatch.setattr(worker.shutil,'disk_usage',lambda path:SimpleNamespace(free=1024**4))
    r=worker.evaluate(dict(z=[0.],output=str(tmp_path/'result'),until=time.monotonic()+60,target=.02,objective='relay-range',mode='relay-replay',variant=variant,response_aggregation='weakest'))
    assert r['status']=='completed'
    assert r['metrics']['partial_attenuation_passed']==(variant=='nominal')
    assert r['metrics']['relay_response_aggregation']=='weakest'
    if variant=='redundant':np.testing.assert_allclose(r['metrics']['loss'],.25,atol=1e-7)
    assert not r['mediation']['cue_attenuation_reduced']
    assert not r['mediation']['original_body_attenuation_nontrivial']
    assert calls==[(False,False),(True,False),(False,True),(True,True)]


def test_weakest_response_objective_does_not_hide_weak_repeat_with_strong_first_cue():
    data=relay_pair(effect=.0001);parent=data['no_cue']
    for condition in data.values():condition['response'].append(deepcopy(condition['response'][0]))
    base=data['no_cue']['response'][0]['AIYL_release_mean']
    def response(row,delta):
        row['AIYL_release_mean']=base+delta
        row['AIYL_release_half_means']=[base+delta]*2
    response(data['cue']['response'][0],.0001)
    response(data['cue']['response'][1],.00001)
    initial=score(data,parent,.02,1.3,response_aggregation='weakest')
    response(data['cue']['response'][0],.0002)
    unchanged=score(data,parent,.02,1.3,response_aggregation='weakest')
    np.testing.assert_allclose(unchanged['loss'],initial['loss'])
    assert not unchanged['partial_attenuation_passed']
    response(data['cue']['response'][1],.0000261)
    improved=score(data,parent,.02,1.3,response_aggregation='weakest')
    assert improved['loss']<unchanged['loss'] and improved['training_margin_met']
