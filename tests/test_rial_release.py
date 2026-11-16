from copy import deepcopy
import json
import sys

import numpy as np
import pytest
from scipy.special import logit

from r_elegans.rial_release import features, score, interpolated_frequency
from test_rial_sensory import paired


def neural_pair(effect=0., phase=0., permanent=False):
    _, traces, ids = paired(effect=0.)
    result = {}
    for condition, trace in traces.items():
        t = trace['dense_time_s']; gain = np.ones(len(t))
        cue = condition == 'cue'
        if cue:
            gain[(t >= 41) & (t < (70 if permanent else 46))] = 1-effect
        angle = 2*np.pi*1.8*t + np.where((t >= 41) & cue, phase, 0.)
        trace['dense_voltage_mV'][:, 0] = -40+5*logit(.35+.1*gain*np.sin(angle))
        result[condition] = features(trace, ids, 'nominal', -40., 5.)
    return result


def test_release_objective_rejects_phase_and_requires_recovery():
    from scripts.audit_rial_sensory import release_failures
    parent = neural_pair()['no_cue']
    positive = score(neural_pair(effect=.03), parent, .02, 1.3)
    assert positive['training_margin_met']
    assert positive['body_response_amplitude_ratios'][0] > .999
    assert not positive['full_start_stop_qualified']
    assert not release_failures(neural_pair(effect=.03),parent,.02)
    assert any(r['window']=='cue/response/0' for r in release_failures(neural_pair(),parent,.02))
    assert not score(neural_pair(phase=.7), parent, .02)['partial_attenuation_passed']
    assert not score(neural_pair(effect=.2, permanent=True), parent, .02)['partial_attenuation_passed']
    weak = deepcopy(neural_pair(effect=.03))
    for condition in weak.values():
        for category in ('active','response','late'):
            for row in condition[category]: row['RIAL_release_cycle_excursion'] *= .5
    assert not score(weak,parent,.02)['partial_attenuation_passed']
    losses = [score(neural_pair(effect=e),parent,.02,1.3)['loss'] for e in (0.,.005,.01,.02)]
    assert all(a > b for a,b in zip(losses,losses[1:]))


def test_interpolated_frequency_resolves_boundary_without_widening_gate():
    for dt in (.002,.001):
        t = np.arange(30,40,dt)
        for f in (1.9785415,1.9873375):
            measured = interpolated_frequency(t,np.sin(2*np.pi*f*t),.1)
            assert abs(measured-f) < 2e-6
            assert (measured <= 1.98) == (f <= 1.98)


@pytest.mark.parametrize('audit_timeout',[False,True])
def test_three_hour_budget_includes_audit_and_failure_receipt(tmp_path,monkeypatch,audit_timeout):
    import scripts.total_job_budget as runner
    out = tmp_path/'output'; calls=[]
    def run(cmd, seconds):
        calls.append((cmd,seconds))
        if 'scripts.audit_rial_sensory' in cmd:
            folder = tmp_path/'output-audit';folder.mkdir()
            if not audit_timeout:(folder/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
            return (124,True) if audit_timeout else (0,False)
        else:
            (out/'summary.json').write_text(json.dumps(dict(execution_valid=True,validation_complete=True)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','rial-release','--output-dir',str(out),
                                   '--source-run',str(tmp_path/'source'),'--total-minutes','180'])
    assert runner.main() == (124 if audit_timeout else 0)
    assert len(calls)==2
    assert '--objective' in calls[0][0] and 'rial-release' in calls[0][0]
    receipt = json.loads((out/'budget_receipt.json').read_text())
    assert receipt['total_minutes']==180 and receipt['all_steps_finished']==(not audit_timeout)
    assert receipt['audit_complete']==(not audit_timeout)
    assert receipt['audit_integrity_passed']==(not audit_timeout)
    assert calls[0][1] < 160*60 and calls[1][1] <= 180*60
    assert json.loads((out/'deadline.json').read_text())['experiment']=='E086'
