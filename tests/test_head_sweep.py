"""Head-relative scans must not be rigid rotation or whole-body undulation."""
import numpy as np
import pytest

from r_elegans.rl.head_sweep import head_metrics,head_checks,assess_head_sweep,relative_head_geometry


def scan(*,whole_body=False,rigid=False,moving=False):
    n=20;t=np.arange(n*10+1)/n;wave=np.sin(2*np.pi*t)
    q=np.zeros((len(t),11))
    if whole_body:q[:]=.03*wave[:,None]
    elif not rigid:q[:,-3:]=wave[:,None]*np.array([.005,.02,.04])
    heading=.2*wave if rigid else np.zeros(len(t))
    position=np.c_[.002*t if moving else np.zeros(len(t)),np.zeros(len(t))]
    return head_metrics(position,heading,q,period_steps=n)


def rest():
    n=201
    return head_metrics(np.zeros((n,2)),np.zeros(n),np.zeros((n,11)),period_steps=20)


def test_relative_scan_passes_and_rigid_body_or_global_bending_does_not():
    good=scan();quiet=rest()
    assert all(head_checks(good,good).values())
    assert assess_head_sweep(good,good,quiet,good,good)['passed']
    for bad in (scan(whole_body=True),scan(rigid=True),scan(moving=True)):
        assert not assess_head_sweep(bad,bad,quiet,bad,good)['passed']
    assert scan(rigid=True)['head_angle_p2p']==0


def test_head_scan_requires_quiet_rest_and_timestep_consistency():
    good=scan();quiet=rest()
    assert not assess_head_sweep(good,good,good,good,good)['passed']
    changed=good|dict(head_angle_p2p=good['head_angle_p2p']*3)
    assert not assess_head_sweep(changed,good,quiet,good,good)['passed']
    static=quiet|dict(mean_forward_body_lengths=.002)
    assert not assess_head_sweep(static,static,quiet,good,good)['passed']


def test_anterior_target_is_causal_localized_and_physically_feasible():
    import jax
    import jax.numpy as jnp
    from r_elegans.body import initialize_muscle_body
    from r_elegans.body.neuromuscular import muscle_longitudinal_positions
    from r_elegans.model import load_builtin_model
    from r_elegans.rl.motor_curriculum import muscle_sequence_body
    from r_elegans.rl.head_sweep import head_target,head_stimulation_vector
    commands=jnp.zeros(1200).at[120:].set(.85*jnp.sin(2*jnp.pi*jnp.arange(1080)/120))
    target=head_target(commands)
    np.testing.assert_allclose(target[:120],.4,atol=1e-7)
    posterior=np.asarray(muscle_longitudinal_positions())>.5
    assert np.max(np.abs(np.asarray(target)[:,posterior]-.4))<.00002
    body=jax.jit(muscle_sequence_body)(target)
    s=jax.tree_util.tree_map(lambda a,b:np.concatenate([np.asarray(a)[None],np.asarray(b)]),initialize_muscle_body(12),body)
    report=head_metrics(s.position,s.heading,s.joint_angles,period_steps=120)
    assert all(head_checks(report,report).values())
    model=load_builtin_model();stim=head_stimulation_vector(model)
    outside=np.asarray(model.neuromuscular_params.synapse_weights.sum(axis=0))==0
    np.testing.assert_array_equal(np.asarray(stim)[outside],0)
    assert np.linalg.norm(stim)>0
