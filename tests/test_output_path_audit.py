import numpy as np
import pytest

from r_elegans.output_path_audit import analyze, projection, spectral_balance


def test_antagonist_cancellation_and_opposite_phase():
    t = np.linspace(0, 10, 2001)
    x = np.sin(2*np.pi*2*t)[:, None]*np.ones((1, 11))
    assert np.max(spectral_balance(t, x, x, 2)['survival_ratio']) < 1e-14
    np.testing.assert_allclose(spectral_balance(t, x, -x, 2)['survival_ratio'], 1)
    assert spectral_balance(t, x, x, None)['survival_ratio'] is None


def test_static_and_wrong_projection_are_not_motion():
    t = np.linspace(0, 60, 6001)
    muscles = np.full((len(t), 95), .5)
    d, v = projection()
    trace = dict(dense_time_s=t, dense_muscles=muscles,
                 dense_command=muscles @ (d-v).T, dense_filtered_drive=np.zeros((len(t), 11)),
                 dense_joints=np.zeros((len(t), 11)), dense_voltage_mV=np.zeros((len(t), 302)))
    result = analyze(trace, ['RMDR']+[str(i) for i in range(301)], np.zeros(95))
    assert result['minimum_joint_cycles_tail_middle_head'] == [0, 0, 0]
    trace['dense_command'] += .001
    with pytest.raises(ValueError, match='projection disagrees'):
        analyze(trace, ['RMDR']+[str(i) for i in range(301)], np.zeros(95))
