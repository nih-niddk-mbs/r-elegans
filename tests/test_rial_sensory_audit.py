import numpy as np

from scripts.audit_rial_sensory import gait_failures, signal_windows


def test_failed_gait_gate_names_region_and_window():
    def row(frequency):
        return dict(regional_cycles_rad=[.01]*3, frequency_hz=frequency)
    parent = dict(active=[row([1.8]*3)])
    paired = {k:dict(active=[row([1.8]*3)], response=[row([1.8]*3)], late=[row([1.8]*3)])
              for k in ('no_cue','cue')}
    assert gait_failures(paired,parent) == []
    paired['cue']['response'][0]['frequency_hz'][1] = 2.1
    failed = gait_failures(paired,parent)
    assert {r['check'] for r in failed} == {'absolute_frequency','relative_frequency'}
    assert all(r['region']==1 and r['window']=='cue/response/0' for r in failed)


def test_signal_audit_release_and_missing_stages_are_explicit():
    a = dict(dense_time_s=np.arange(4.), dense_voltage_mV=np.zeros((4,1)))
    b = dict(dense_time_s=np.arange(4.), dense_voltage_mV=np.ones((4,1)))
    result = signal_windows(a,b,['RIAL'],np.array([0.]),np.array([1.]),[(0,4)])[0]
    assert result['stages']['dense_joints']['missing']
    release = result['release_by_neuron']['RIAL']
    np.testing.assert_allclose(release['no_cue_mean'],.5)
    np.testing.assert_allclose(release['difference_rms'],1/(1+np.exp(-1))-.5)
    assert result['stages']['dense_voltage_mV']['cue_minus_sham_rms'] == [1.]
