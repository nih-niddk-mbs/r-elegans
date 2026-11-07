import jax
import jax.numpy as jnp
import numpy as np
import pytest

from r_elegans.brain.conductance_fitting import (
    voltage_clamp_trace, fit_clamp_conductances, sensitivity_diagnostics,
    synaptic_design, fit_synaptic_conductances,
)
from r_elegans.brain.single_compartment import CHANNEL_NAMES, CHANNEL_INDEX
from r_elegans.data import load_neuron_parameters


def test_voltage_clamp_physical_units_ohmic_limit_and_gradients():
    base = load_neuron_parameters("AWCON").params
    conductance = jnp.zeros(len(CHANNEL_NAMES)).at[CHANNEL_INDEX["LEAK"]].set(.4)
    params = base._replace(conductances_nS=conductance)
    voltage = jnp.array([-60.,-30.,0.,30.])
    result = jax.jit(lambda p: voltage_clamp_trace(p,voltage,.4))(params)
    np.testing.assert_allclose(result,.4*(np.asarray(voltage)+90.),atol=1e-5)
    gradient = jax.grad(lambda g: jnp.sum(voltage_clamp_trace(
        params._replace(conductances_nS=g),voltage,.4)))(conductance)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient[CHANNEL_INDEX["LEAK"]],300.,atol=1e-4)


def test_maximal_conductance_recovery_in_identifiable_ohmic_limit():
    voltage = jnp.array([-80.,-50.,-20.,10.])
    def predict(g):
        return g[0]*(voltage+90.) + g[1]*(voltage-30.)
    truth = jnp.array([.4,.15])
    fitted, report = fit_clamp_conductances(predict,np.asarray(predict(truth)),
        [.1,.05],log_prior_weight=0.,max_evaluations=50)
    assert report["optimizer_success"]
    np.testing.assert_allclose(fitted,truth,rtol=1e-4,atol=1e-5)
    assert report["sensitivity"]["rank"] == 2


def test_synaptic_fit_recovers_chemical_and_shared_gap_with_heldout_stimuli():
    from scipy.special import expit
    rng = np.random.default_rng(5)
    voltage = rng.uniform(-65.,-5.,(150,2))
    # Independently evaluate a 0 -> 1 excitatory synapse plus a shared gap.
    current = np.zeros_like(voltage)
    current[:,1] = .3*expit((voltage[:,0]+35.)/6.)*voltage[:,1]
    gap_current = .12*(voltage[:,0]-voltage[:,1])
    current[:,0] += gap_current
    current[:,1] -= gap_current
    chemical = np.array([[0,0],[1,0]])
    gap = np.array([[0,1],[1,0]])
    reversal = np.zeros((2,2))
    threshold, slope = np.full(2,-35.), np.full(2,6.)
    train, edges = synaptic_design(voltage[:100],chemical,gap,reversal,threshold,slope)
    test, _ = synaptic_design(voltage[100:],chemical,gap,reversal,threshold,slope)
    g, report = fit_synaptic_conductances(train,current[:100])
    assert edges == [("chemical",1,0),("gap",0,1)]
    assert report["optimizer_success"]
    np.testing.assert_allclose(g,[.3,.12],atol=1e-7)
    np.testing.assert_allclose(test@g,current[100:].ravel(),atol=1e-6)


def test_rank_deficiency_and_unknown_synaptic_polarity_are_not_hidden():
    diagnostics = sensitivity_diagnostics(np.array([[1.,1.],[2.,2.],[3.,3.]]))
    assert diagnostics["rank"] == 1 and diagnostics["condition_number"] is None
    chemical = np.array([[0,0],[1,0]])
    with pytest.raises(ValueError,match="reversals"):
        synaptic_design(np.zeros((5,2)),chemical,np.zeros((2,2)),
            np.full((2,2),np.nan),np.zeros(2),np.ones(2))
    with pytest.raises(ValueError,match="symmetric"):
        synaptic_design(np.zeros((5,2)),chemical,chemical,np.zeros((2,2)),np.zeros(2),np.ones(2))
