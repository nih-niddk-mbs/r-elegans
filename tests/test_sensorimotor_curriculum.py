import numpy as np
import pytest
from r_elegans.sensorimotor_curriculum import ResponseProtocol,cue_world,response_metrics

IDS=('AWAL','AWAR')


def traces():
    t=np.linspace(0,8,801)
    sham=dict(dense_time_s=t,dense_joints=np.zeros((len(t),11)),dense_voltage_mV=np.full((len(t),2),-40.))
    active={k:v.copy() for k,v in sham.items()}
    pulse=np.where((t>=2)&(t<4),.004*np.sin(np.pi*(t-2)/2),0.)
    active['dense_joints'][:,5]=pulse
    active['dense_voltage_mV'][(t>=2)&(t<3)]+=.2
    return active,sham


def test_reversible_evoked_response_and_reject_posture_spontaneous_or_nonrecovering():
    a,b=traces()
    result=response_metrics(a,b,IDS)
    assert result['passed'] and result['selected_joint']==5
    assert not response_metrics(a,a,IDS)['passed']  # Shared spontaneous movement is no cue response.
    a['dense_joints'][a['dense_time_s']>=2,5]=.004
    result=response_metrics(a,b,IDS)
    assert not result['passed'] and result['recovery_ratio']==pytest.approx(1.)
    a['dense_joints'][:]=.004
    assert not response_metrics(a,b,IDS)['passed']  # Static/pre-cue differences cannot earn response.


def test_recruitment_and_motor_recovery_must_both_pass_and_grids_match():
    a,b=traces()
    a['dense_voltage_mV'][a['dense_time_s']>=2]+=.2
    assert not response_metrics(a,b,IDS)['passed']  # Sensory activity must recover too.
    a['dense_voltage_mV'][:]=-40.
    assert not response_metrics(a,b,IDS)['passed']
    a['dense_time_s'][1]+=.001
    with pytest.raises(ValueError,match='grids'):response_metrics(a,b,IDS)
    with pytest.raises(ValueError):ResponseProtocol(onset_s=7.)


def test_only_environmental_diacetyl_changes_and_washout_is_finite():
    import jax.numpy as jnp
    from r_elegans.body.embodied_sensory import sample_chemical_world
    world=cue_world();sham=cue_world(sham=True)
    head=jnp.zeros(2)
    for t in (0.,1.,3.,7.):
        np.testing.assert_array_equal(sample_chemical_world(head,t,world),sample_chemical_world(head,t,sham))
    delta=np.asarray(sample_chemical_world(head,2.5,world)-sample_chemical_world(head,2.5,sham))
    np.testing.assert_array_equal(delta,[1.,0.,0.])
    with pytest.raises(ValueError):cue_world(onset_shift=8.)


def test_worst_case_selection_keeps_weak_cue_visible_after_nominal_success():
    from r_elegans.sensorimotor_curriculum import training_objective
    assert training_objective([dict(total=0.)])==0.
    assert training_objective([dict(total=0.),dict(total=.2)],'nominal-low-dose')==.2
    assert training_objective([dict(total=.05),dict(total=.1)],'nominal-low-dose')==.1
    with pytest.raises(ValueError):training_objective([dict(total=.2)],'nominal-low-dose')
    with pytest.raises(ValueError):training_objective([dict(total=float('nan'))])
