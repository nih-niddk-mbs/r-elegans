"""Steering must turn the moving centroid, not just the head/body pose."""
import numpy as np
import pytest

from r_elegans.rl.steering import steering_metrics,assess_steering,selection_key


def trajectory(turn=.002,progress=.002,*,heading_only=False):
    n=20;t=np.arange(n*10+1)/n;theta=turn*t
    if turn and not heading_only:
        radius=progress/turn
        position=np.c_[radius*np.sin(theta),radius*(1-np.cos(theta))]
    else:
        position=np.c_[progress*t,np.zeros(len(t))]
    angles=.02*np.sin(2*np.pi*t)[:,None]*np.sin(np.linspace(0,np.pi,11))
    return steering_metrics(position,theta,angles,period_steps=n)


@pytest.mark.parametrize('sign',[-1,1])
def test_steering_accepts_real_arcs_and_rejects_pose_only_or_wrong_direction(sign):
    arc=trajectory(turn=sign*.002);rest=trajectory(turn=0,progress=0)
    assert assess_steering(arc,arc,rest,rest,arc,sign=sign)['passed']
    assert not assess_steering(arc,arc,rest,rest,arc,sign=-sign)['passed']
    pose=trajectory(turn=sign*.002,heading_only=True)
    assert not assess_steering(pose,pose,rest,rest,arc,sign=sign)['passed']
    stationary=trajectory(turn=sign*.002,progress=0)
    assert not assess_steering(stationary,stationary,rest,rest,arc,sign=sign)['passed']


def test_steering_keeps_progress_and_numerical_control_floors():
    arc=trajectory();rest=trajectory(turn=0,progress=0)
    assert not assess_steering(arc,trajectory(turn=.008),rest,rest,arc,sign=1)['passed']
    assert not assess_steering(arc,arc,arc,rest,arc,sign=1)['passed']
    assert not assess_steering(arc,arc,rest,arc,arc,sign=1)['passed']
    weak=trajectory(progress=.0005)
    assert not assess_steering(weak,weak,rest,rest,arc,sign=1)['passed']


def test_selection_prefers_physical_turn_over_better_imitation_alone():
    m=dict(muscle_rmse=.005,body_relative_rmse=.1,bend_correlation=.99,direction_agreement=1.,
        muscle_lag=.14,target_muscle_lag=.14,body_lag=.14,target_body_lag=.14)
    arc=trajectory();straight=trajectory(turn=0)
    perfect=m|dict(muscle_rmse=0.,body_relative_rmse=0.,bend_correlation=1.)
    assert selection_key([m],arc,arc,sign=1)<selection_key([perfect],straight,arc,sign=1)


def test_biased_targets_drive_opposite_physical_turns():
    import jax
    import jax.numpy as jnp
    from r_elegans.body import initialize_muscle_body
    from r_elegans.rl.motor_curriculum import muscle_sequence_body
    from r_elegans.rl.steering import steering_target
    n=120;c=.85*jnp.sin(2*jnp.pi*jnp.arange(n*10)/n)
    for sign in (-1,1):
        body=jax.jit(lambda c:muscle_sequence_body(steering_target(c,bias=sign*.006)))(c)
        s=jax.tree_util.tree_map(lambda a,b:np.concatenate([np.asarray(a)[None],np.asarray(b)]),initialize_muscle_body(12),body)
        m=steering_metrics(s.position,s.heading,s.joint_angles,period_steps=n)
        assert m['mean_forward_body_lengths']>.001
        assert sign*m['mean_signed_yaw_per_cycle']>5e-5
        assert sign*m['mean_path_turn_per_cycle']>5e-5


def test_body_trajectory_objective_has_finite_useful_gradients():
    import jax
    import jax.numpy as jnp
    from r_elegans.rl.motor_curriculum import muscle_sequence_body
    from r_elegans.rl.steering import steering_target,trajectory_objective
    commands=jnp.sin(jnp.arange(32)*.1)
    target=steering_target(commands)[None]
    body=jax.vmap(muscle_sequence_body)(target)
    loss=jax.jit(lambda predicted:trajectory_objective(predicted,target,body))
    assert float(loss(target))==pytest.approx(0.,abs=1e-10)
    wrong=.8-target
    value,gradient=jax.jit(jax.value_and_grad(loss))(wrong)
    assert np.isfinite(float(value)) and np.all(np.isfinite(gradient))
    assert float(jnp.linalg.norm(gradient))>0
    corrected=wrong-1e-4*gradient/jnp.maximum(jnp.linalg.norm(gradient),1)
    assert float(loss(corrected))<float(value)
