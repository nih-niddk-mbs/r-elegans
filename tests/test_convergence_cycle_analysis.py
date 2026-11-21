import numpy as np
import pytest

from scripts.analyze_convergence_cycles import window_stats, descriptive_comparison


def test_interpolated_full_train_recovers_incommensurate_period():
    dt, period = .002, .23317
    t = (np.arange(20000) + 1) * dt
    v = -20 + 10 * np.sin(2 * np.pi * t / period + .3)
    result = window_stats(v, dt, 20, 40)
    assert result['available']
    assert abs(result['whole_train_period_s'] - period) < 1e-7
    assert abs(result['last_five_interpolated_mean_period_s'] - period) < 1e-6
    assert result['period_cv'] < 1e-4
    assert len(result['mean_cycle_voltage_mV']) == 101
    assert all(20 <= x < 40 for x in result['peak_times_sampled_s'])


def test_window_statistics_reveal_changing_period():
    dt = .002
    t = (np.arange(20000) + 1) * dt
    v = 10 * np.sin(2 * np.pi * (4 * t + .01 * t ** 2))
    early = window_stats(v, dt, 20, 25)
    late = window_stats(v, dt, 35, 40)
    whole = window_stats(v, dt, 20, 40)
    assert early['whole_train_period_s'] > late['whole_train_period_s']
    assert whole['period_cv'] > .01
    assert abs(whole['whole_train_period_s'] - whole['last_five_interpolated_mean_period_s']) > .001
    comparison = descriptive_comparison(early, late)
    assert 'passed' not in comparison  # analysis must not invent replacement acceptance gates


def test_plateau_is_not_a_cycle_and_bad_inputs_are_rejected():
    result = window_stats(np.full(20000, -30.), .002, 20, 40)
    assert not result['available']
    assert result['peaks'] == 0
    assert not descriptive_comparison(result, result)['available']
    with pytest.raises(ValueError):
        window_stats(np.full(20, np.nan), .002, 0, 1)
    with pytest.raises(ValueError):
        window_stats(np.zeros(20), 0, 0, 1)
