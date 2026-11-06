import hashlib
import json

import numpy as np
import pytest

from r_elegans.rl.bidirectional_curriculum import command_bank, validation_key
from scripts.run_motor_curriculum import refinement_checkpoints, find_refinement_run


def metric(reverse=False, error=.05):
    lag = -.14 if reverse else .14
    return dict(muscle_rmse=.005, body_relative_rmse=error, bend_correlation=.99,
        direction_agreement=1., muscle_lag=lag, target_muscle_lag=lag,
        body_lag=lag, target_body_lag=lag)


def test_broader_commands_are_reproducible_and_tuning_is_independent():
    train = command_bank()
    tuning = command_bank(768, held_out=True)
    assert train.shape == (8,384) and tuning.shape == (8,768)
    np.testing.assert_array_equal(train,command_bank())
    assert np.all(np.isfinite(train)) and np.max(np.abs(train)) <= 1
    assert not np.allclose(train,tuning[:,:384])
    assert np.count_nonzero(train == 0) > 300


def test_joint_selection_rejects_forward_forgetting_and_wrong_reverse_lags():
    good = validation_key([metric()], [metric(True)])
    assert good[0] == 0
    forgot_forward = validation_key([metric(error=.5)], [metric(True,error=.001)])
    assert good < forgot_forward
    wrong = metric(True) | dict(muscle_lag=.14,body_lag=.14)
    assert good < validation_key([metric()],[wrong])
    assert validation_key([metric(error=.02)],[metric(True,error=.02)]) < good
    with pytest.raises(ValueError):
        validation_key([metric()],[metric(True) | dict(muscle_rmse=float('nan'))])


def test_tuning_targets_have_valid_propagation_in_both_directions():
    import jax
    import jax.numpy as jnp
    from r_elegans.rl.motor_curriculum import traveling_bend_target, traveling_metrics, traveling_passes
    commands = jnp.asarray(command_bank(768,held_out=True))
    evaluate = jax.jit(jax.vmap(traveling_metrics))
    for reverse in (False,True):
        target = jax.vmap(lambda c: traveling_bend_target(c,amplitude=.07,reverse=reverse))(commands)
        result = evaluate(target,target)
        for i in range(len(commands)):
            m = {k:float(v[i]) for k,v in result.items()}
            assert traveling_passes(m,reverse=reverse), (i,reverse,m)


def source_cohort(tmp_path):
    reference = tmp_path / "original/actor_best.npz"
    reference.parent.mkdir()
    reference.write_bytes(b"original forward actor")
    root = tmp_path / "reverse-0001"
    actor = root / "seed-0/reverse/actor_best.npz"
    actor.parent.mkdir(parents=True)
    actor.write_bytes(b"bidirectional candidate")
    (actor.parent/"manifest.json").write_text(json.dumps(dict(initial_actor=str(reference),
        initial_actor_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())))
    complete = dict(status="completed",error=None,exit_code=2,passed=False,control_passed=True)
    summary = dict(workers=[dict(seed=0,stages={name:dict(complete)
        for name in ("reverse","forward_retention","reverse_locomotion")})])
    (root/"summary.json").write_text(json.dumps(summary))
    return root,actor,reference,summary


def test_refinement_can_use_failed_readiness_without_promoting_or_losing_original_teacher(tmp_path):
    from scripts.refine_reverse import forward_reference
    root,actor,reference,summary = source_cohort(tmp_path)
    assert refinement_checkpoints(root,[0])[0] == actor
    assert find_refinement_run(tmp_path,[0])[0] == root
    assert forward_reference(actor) == reference
    with pytest.raises(ValueError):
        refinement_checkpoints(root,[1])
    reference.write_bytes(b"modified source")
    with pytest.raises(ValueError):
        refinement_checkpoints(root,[0])
    with pytest.raises(ValueError):
        forward_reference(actor)


def test_refinement_rejects_crashes_or_loss_of_motor_control(tmp_path):
    root,actor,reference,summary = source_cohort(tmp_path)
    summary['workers'][0]['stages']['reverse_locomotion']['control_passed'] = False
    (root/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        refinement_checkpoints(root,[0])
