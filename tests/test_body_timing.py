import numpy as np
from r_elegans.body_timing import command_indices, refinement_check


def test_retiming_holds_each_source_step_and_refines_clock():
    np.testing.assert_array_equal(command_indices(4,.002,1,.002,5),[1,2,3,4])
    np.testing.assert_array_equal(command_indices(8,.002,.5,.002,5),[1,1,2,2,3,3,4,4])
    np.testing.assert_array_equal(command_indices(8,.001,1,.002,5),[1,1,2,2,3,3,4,4])
    np.testing.assert_array_equal(command_indices(8,.001,1,.002-1e-15,5),[1,1,2,2,3,3,4,4])


def test_refinement_does_not_accept_different_drift_when_cycle_counts_are_zero():
    t=np.linspace(0,4,801)
    def trace(slope):
        q=slope*t[:,None]*np.ones((1,11))
        return dict(time_s=t,joints=q,command=q,drive=q,force_residual=np.zeros(800),
                    below_drag_regularization_fraction=np.zeros(800),
                    **{key:np.zeros((800,11)) for key in ('active_moment_N_m','elastic_moment_N_m',
                         'internal_viscous_moment_N_m','inferred_environmental_moment_N_m')})
    check=refinement_check(trace(.001),trace(.002))
    assert check['supported_cycle_joints']==0
    assert check['cycles_passed']
    assert not check['waveform_passed']
    assert not check['passed']
