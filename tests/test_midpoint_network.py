from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from r_elegans.brain.conductance_network import load_network,initial_network_state,advance_network
from r_elegans.brain.midpoint_network import advance_midpoint
from r_elegans.brain.midpoint_replay import cycle_phases


def linear_network(gap=.1):
    root=Path(__file__).resolve().parents[1]
    p,f,c=load_network(root/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz')
    cells=jax.tree.map(lambda a:jnp.asarray(a[:2],dtype=jnp.float64),f.cells)
    cells=cells._replace(conductances_nS=jnp.zeros((2,17)).at[:,16].set(.03),capacitance_pF=jnp.ones(2),leak_reversal_mV=jnp.full(2,-55.))
    mask=jnp.array([[0.,1.],[1.,0.]])
    p=p._replace(log_channel_scale=jnp.zeros((2,17)),raw_chemical=jnp.zeros((2,2)),raw_gap=jnp.full((2,2),np.log(np.expm1(gap/c.synaptic_unit_nS))) if gap else jnp.zeros((2,2)))
    f=f._replace(cells=cells,chemical_mask=jnp.zeros((2,2)),gap_mask=mask if gap else jnp.zeros((2,2)),
        synapse_reversal_mV=jnp.zeros(2),release_threshold_mV=jnp.full(2,-35.),release_slope_mV=jnp.full(2,5.),bias_pA=jnp.zeros(2),initial_voltage_mV=jnp.array([-70.,-40.]))
    return p,f,c,initial_network_state(f,c)


def test_gap_coupled_analytic_solution_shows_second_order_convergence():
    p,f,c,s=linear_network()
    exact=-55.+np.array([-15.,15.])*np.exp(-.23*10.)
    errors={}
    for method,advance in [('legacy',advance_network),('midpoint',advance_midpoint)]:
        e=[]
        for n in (20,40,80):
            result=jax.jit(lambda:advance(s,p,f,.01,n,c))()
            e.append(float(np.max(abs(np.asarray(result.voltage_mV)-exact))))
        errors[method]=e
    assert errors['midpoint'][0]/errors['midpoint'][1]>3.5
    assert errors['midpoint'][1]/errors['midpoint'][2]>3.5
    assert 1.8<errors['legacy'][1]/errors['legacy'][2]<2.2
    assert errors['midpoint'][0]<errors['legacy'][0]/10


def test_isolated_leak_remains_exact_and_parameters_unchanged():
    p,f,c,s=linear_network(gap=0.)
    before=[np.asarray(a).copy() for a in jax.tree.leaves((p,f))]
    result=jax.jit(lambda:advance_midpoint(s,p,f,.01,20,c))()
    exact=-55.+np.array([-15.,15.])*np.exp(-.03*10.)
    np.testing.assert_allclose(result.voltage_mV,exact,rtol=0,atol=1e-12)
    assert np.isfinite(np.asarray(result.gates)).all()
    for old,new in zip(before,jax.tree.leaves((p,f))):np.testing.assert_array_equal(old,new)


def test_multicycle_sampling_spans_recording_and_preserves_cycle_groups():
    t=(np.arange(4000)+1)*.001
    ix,m=cycle_phases(-30+10*np.cos(2*np.pi*t/.23),.001)
    assert len(ix)==16 and len(set(ix))==16 and np.all(np.diff(ix)>0)
    assert ix[-1]-ix[0]>3000
    assert len(m['selected_cycles'])==4 and m['complete_cycles']>=16
    with pytest.raises(ValueError):cycle_phases(np.zeros(4000),.001)


def test_midpoint_replay_supports16_phases_and_sham_only_traces(monkeypatch):
    from types import SimpleNamespace
    import r_elegans.brain.midpoint_network as midpoint
    import r_elegans.brain.extrasynaptic as extra
    from r_elegans.brain.midpoint_replay import make_replay
    from r_elegans.brain.peptide_state_replay import FullState
    from r_elegans.brain.conductance_network import ConductanceNetworkState
    def advance(s,p,f,dt,sub,c,membrane_input):
        return s._replace(voltage_mV=s.voltage_mV+dt*membrane_input.conductance_nS,indicator=s.indicator+dt*membrane_input.conductance_nS)
    monkeypatch.setattr(midpoint,'advance_midpoint',advance)
    monkeypatch.setattr(extra,'peptide_membrane_input',lambda s,e,i,u:SimpleNamespace(conductance_nS=jnp.zeros_like(s),weighted_reversal_pA=jnp.zeros_like(s)))
    monkeypatch.setattr(extra,'peptide_level_step',lambda level,*a:level)
    start=FullState(ConductanceNetworkState(jnp.zeros((16,2)),jnp.zeros((16,2,35)),jnp.zeros((16,2)),jnp.zeros((16,2))),jnp.zeros((16,2)))
    stimuli=jnp.array([[.03,0.],[0.,.03],[.03,.03],[0.,0.],[0.,0.]])
    pp=extra.PeptideParams(jnp.zeros(1),jnp.zeros(1),jnp.array(0.))
    kernel=make_replay(SimpleNamespace(synaptic_unit_nS=1.),[0,1],.00025,.0078125,seconds=.004,stimulus_seconds=.002,window=(.001,.003),trace_mode='shams')
    means,final,traces=jax.tree.map(np.asarray,kernel(None,None,pp,None,None,start,stimuli))
    assert means.shape==(16,5,2) and traces[0].shape==(16,2,4,2)
    assert traces[4].shape==(16,5,4)
    np.testing.assert_array_equal(traces[0],np.zeros((16,2,4,2)))
    assert means[0,0,0]>0 and final.neural.voltage_mV[0,0,0]>0
