import importlib.util
import pytest
if importlib.util.find_spec("gymnax") is None:
    pytest.skip("gymnax extra not installed", allow_module_level=True)

import numpy as np
from r_elegans.rl.locomotion import cycle_metrics, assess_propulsion
from scripts.run_motor_curriculum import baseline_valid, resume_checkpoints, find_resume_run


def trajectory(progress=.002, startup=0., drift_shape=False, rotated=False):
    samples = np.arange(201)
    phase = 2 * np.pi * samples / 20
    angles = .03 * np.sin(phase)[:, None] * np.sin(np.linspace(0, np.pi, 11))[None, :]
    if drift_shape:
        angles += .0005 * samples[:, None]
    distance = progress * samples / 20 + startup * (1 - np.exp(-samples / 2))
    position = np.stack([distance, np.zeros_like(distance)], axis=1)
    heading = np.zeros_like(distance)
    if rotated:
        position = position[:, ::-1]
        heading += np.pi / 2
    return cycle_metrics(position, heading, angles, period_steps=20)


def test_phase_matched_progress_discards_startup_and_uses_body_axis():
    normal = trajectory(startup=.5)
    rotated = trajectory(startup=.5, rotated=True)
    assert normal["mean_forward_body_lengths"] == pytest.approx(.002, abs=1e-9)
    assert rotated["mean_forward_body_lengths"] == pytest.approx(.002, abs=1e-9)
    assert abs(rotated["mean_lateral_body_lengths"]) < 1e-9
    assert normal["max_shape_closure_relative"] < 1e-10
    # A transient centroid shift alone is not sustained locomotion.
    startup = trajectory(progress=0, startup=.5)
    assert not assess_propulsion(startup, startup, trajectory(progress=0))["detectable_translation"]


def test_propulsion_requires_periodicity_numerical_agreement_and_useful_size():
    steady = trajectory()
    rest = trajectory(progress=0)
    assert assess_propulsion(steady, steady, rest)["passed"]
    weak = trajectory(progress=.0003)
    result = assess_propulsion(weak, weak, rest)
    assert result["detectable_translation"]
    assert not result["passed"]
    assert not assess_propulsion(steady, trajectory(progress=-.002), rest)["passed"]
    assert not assess_propulsion(steady, trajectory(progress=.01), rest)["passed"]
    assert not assess_propulsion(steady, steady, steady)["passed"]
    assert not assess_propulsion(steady, steady, rest, isotropic=steady)["detectable_translation"]
    drifting = trajectory(drift_shape=True)
    assert not assess_propulsion(drifting, drifting, rest)["passed"]


def test_forward_axis_comes_from_shape_not_first_rod_heading():
    samples = np.arange(201)
    # Eleven equal bends put the chord axis at 0.11 rad, while rod 0 is at 0.
    distance = .004 * samples / 20
    position = distance[:, None] * np.array([np.cos(.11), np.sin(.11)])
    metrics = cycle_metrics(position, np.zeros(201), np.full((201, 11), .02),
                            period_steps=20, body_length=2.)
    assert metrics["mean_forward_body_lengths"] == pytest.approx(.002, abs=1e-12)
    assert abs(metrics["mean_lateral_body_lengths"]) < 1e-12
    assert np.linalg.norm(metrics["total_displacement_body_lengths"]) == pytest.approx(.012)


def test_wave_resume_does_not_select_an_alternation_checkpoint(tmp_path):
    import json
    folder = tmp_path / "wave-0001/seed-0/wave"
    folder.mkdir(parents=True)
    (folder / "actor_best.npz").write_bytes(b"wave checkpoint")
    root = folder.parent.parent
    (root / "summary.json").write_text(json.dumps(dict(workers=[dict(seed=0,
        stages=dict(wave=dict(passed=True, status="completed", exit_code=0)))])))
    assert resume_checkpoints(root, [0], stage="wave")[0] == folder / "actor_best.npz"
    assert find_resume_run(tmp_path, [0], stage="wave")[0] == root
    with pytest.raises(ValueError):
        resume_checkpoints(root, [0])


def test_only_valid_baseline_can_advance_amplitude():
    weak = dict(status="completed", error=None, exit_code=2, passed=False,
                periods=[dict(detectable_translation=True, approximately_straight=True)] * 3)
    assert baseline_valid(weak)
    assert not baseline_valid(weak | dict(status="failed"))
    assert not baseline_valid(weak | dict(error="numerical failure"))
    assert not baseline_valid(weak | dict(periods=[dict(detectable_translation=False)] * 3))
