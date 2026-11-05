"""Behavior gates reject drift, reciprocal motion and failed recovery."""
import numpy as np
import pytest

from r_elegans.rl.stop_restart import stop_restart_commands, stop_restart_metrics, assess_stop_restart


def reports(*, never_stop=False, no_restart=False, reciprocal_stop=False):
    n = 20
    command = stop_restart_commands(n)
    block = len(command) // 3
    t = np.arange(len(command) + 1)
    active = np.ones(len(command))
    if not never_stop:
        active[block:2*block] = 0
    if no_restart:
        active[2*block:] = 0
    x = np.r_[0., np.cumsum(active * .002 / n)]
    angles = .03 * np.sin(2*np.pi*t/n)[:, None] * np.ones((1, 11))
    angles[block:2*block+1] *= float(never_stop or reciprocal_stop)
    if no_restart:
        angles[2*block:] = 0
    if reciprocal_stop:
        x[block:2*block+1] += .002 * np.sin(2*np.pi*np.arange(block+1)/n)
    return stop_restart_metrics(np.stack([x, x*0], axis=1),
        np.zeros(len(x)), angles, period_steps=n, dt=.02)


def control(report):
    return {k: v | dict(mean_forward_body_lengths=0.) for k, v in report.items()}


def test_continuous_commands_have_quiet_middle_and_restarted_phase():
    c = stop_restart_commands(20)
    assert len(c) == 360
    np.testing.assert_array_equal(c[:120], c[240:])
    assert np.max(np.abs(c[120:240])) == 0
    assert np.max(np.abs(c[:120])) == pytest.approx(.85)


def test_stop_restart_requires_quiescence_and_recovered_forward_motion():
    good = reports()
    assert assess_stop_restart(good, good, control(good))["passed"]
    for bad in (reports(never_stop=True), reports(no_restart=True), reports(reciprocal_stop=True)):
        assert not assess_stop_restart(bad, bad, control(bad))["passed"]
    # A zero net displacement must not conceal ongoing back-and-forth motion.
    reciprocal = reports(reciprocal_stop=True)
    assert abs(reciprocal["stop"]["mean_forward_body_lengths"]) < 1e-10
    assert reciprocal["stop"]["centroid_path_per_cycle"] > .005


def test_stop_restart_keeps_progress_floor_and_timestep_controls():
    good = reports()
    weak = {k: v | dict(mean_forward_body_lengths=v["mean_forward_body_lengths"] * .1)
            for k, v in good.items()}
    result = assess_stop_restart(weak, weak, control(weak))
    assert result["control_passed"]
    assert not result["passed"]
    assert not assess_stop_restart(good, good, good)["passed"]
    different = {k: v | dict(mean_forward_body_lengths=v["mean_forward_body_lengths"] * 3)
                 for k, v in good.items()}
    assert not assess_stop_restart(different, good, control(good))["passed"]
