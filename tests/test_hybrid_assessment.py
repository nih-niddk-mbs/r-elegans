import copy

import numpy as np

from r_elegans.hybrid_assessment import coordination_statistics, cycle_excursions, validation_checks
from r_elegans.rhythm_comparison import hybrid_validation_cases


def test_raw_supported_standing_and_travelling_shapes_differ():
    t = np.linspace(0, 60, 6001)
    standing = .02 * np.sin(2*np.pi*t[:, None]) * np.linspace(.5, 1., 11)[None, :]
    a = coordination_statistics(t, standing)
    assert a['eligible_joints'] == list(range(11))
    assert abs(a['frequency_hz'] - 1) < .01
    assert a['detrended_shape_pc1_fraction'] > .999
    assert a['coherent_phase_span_rad'] < .01
    travelling = .02 * np.sin(2*np.pi*t[:, None] - np.linspace(0, np.pi, 11)[None, :])
    b = coordination_statistics(t, travelling)
    assert abs(b['frequency_hz'] - 1) < .01
    assert b['detrended_shape_pc1_fraction'] < .7
    assert abs(b['coherent_phase_span_rad'] - np.pi) < .03
    drift = np.tile((.1 * np.exp(-t/20))[:, None], (1, 11))
    c = coordination_statistics(t, drift)
    assert c['frequency_hz'] is None and c['phases'] == []


def test_hybrid_controls_are_independent_and_fixed_family():
    cases = {c['name']: c for c in hybrid_validation_cases()}
    assert len(cases) == 8
    assert all(c['channels']=='old' and c['connections']=='new' and c['precision']=='float64' for c in cases.values())
    assert cases['hybrid-body-half']['body_factor'] == .5
    assert cases['hybrid-body-half']['brain_factor'] == 1
    assert cases['hybrid-neural-half'].get('body_factor', 1) == 1
    assert cases['hybrid-held-voltage']['held_voltage']
    assert cases['hybrid-long']['long']


def test_cyclic_excursion_does_not_count_slow_posture_drift_as_large_wiggle():
    t = np.linspace(0, 60, 6001)
    x = .0001 * np.sin(2*np.pi*2*t) + .0002*t
    q = np.tile(x[:, None], (1, 11))
    a = cycle_excursions(q)
    assert np.ptp(x[len(x)//2:]) > .003
    assert 0 < max(a['joint_cycle_excursion_rad']) < .00025
    assert not a['all_three_cycle_amplitude_targets_met']
    drift = np.tile((.1*np.exp(-t/20))[:, None], (1, 11))
    assert cycle_excursions(drift)['joint_cycle_excursion_rad'] == [0.] * 11


def test_same_motor_and_regional_refinement_required_for_rhythm_screen():
    base = dict(hybrid_assessment=dict(rhythm_retained=True,
        motor_profiles={'RMDR': dict(amplitude_mV=10., frequency_hz=2.5)},
        regions=dict(regional_joint_median_p2p_rad=[.004, .02, .01],
                     all_three_regional_amplitude_targets_met=True)),
        late_repeating_by_stage=dict(dense_voltage_mV=1, dense_joints=11))
    results = {c['name']: copy.deepcopy(base) for c in hybrid_validation_cases()}
    results['hybrid-held-voltage']['late_repeating_by_stage'] = dict(dense_voltage_mV=0, dense_joints=0)
    assert validation_checks(results)['rhythm_validation_pass']
    bad = copy.deepcopy(results)
    bad['hybrid-neural-half']['hybrid_assessment']['motor_profiles'] = {'OTHER':dict(amplitude_mV=10., frequency_hz=2.5)}
    assert not validation_checks(bad)['rhythm_validation_pass']
    bad = copy.deepcopy(results)
    bad['hybrid-body-half']['hybrid_assessment']['regions']['regional_joint_median_p2p_rad'][0] *= 2
    assert not validation_checks(bad)['rhythm_validation_pass']
    bad = copy.deepcopy(results)
    bad['hybrid-neural-half']['hybrid_assessment']['motor_profiles']['RMDR']['frequency_hz'] *= 2
    assert not validation_checks(bad)['rhythm_validation_pass']
