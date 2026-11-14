import numpy as np
import pytest

from r_elegans.control_signal_audit import paired_summary,source_currents,replay_chemical


def test_sampled_currents_use_post_pre_orientation_and_inward_sign():
    v=np.array([[-40.,-60.],[-20.,-50.]])
    chemical=np.array([[0.,0.],[2.,0.]])
    gap=np.array([[0.,3.],[3.,0.]])
    # At source threshold, release=.5. Reversal0, target-60 => +60pA.
    c,g=source_currents(v,chemical,gap,np.array([0.,-80.]),np.array([-40.,-60.]),np.ones(2),[0],[1])
    assert c[0,0]==60.
    np.testing.assert_allclose(g[:,0],[60.,90.])
    reverse,_=source_currents(v,chemical,gap,np.array([0.,-80.]),np.zeros(2),np.ones(2),[1],[0])
    np.testing.assert_array_equal(reverse,0.)


def test_paired_summary_keeps_transient_and_sham_distinct():
    t=np.arange(0,90,.01);sham=np.sin(t)[:,None];cue=sham.copy();cue[(t>=40)&(t<42)]+=2
    result=paired_summary(t,cue,sham,['ASEL'])
    assert result['pre'][0]['maximum_absolute_difference']==0
    assert result['stop_cue'][0]['maximum_absolute_difference']==pytest.approx(2.)
    assert result['rest'][0]['maximum_absolute_difference']==0
    with pytest.raises(ValueError):paired_summary(t,cue[:-1],sham,['ASEL'])


def test_offline_replay_matches_actual_chemical_transduction_memory():
    from types import SimpleNamespace
    import jax
    from r_elegans.body.embodied_sensory import (EmbodiedSensoryConfig,build_sensory_maps,
        initial_embodied_sensory_state,transduce_sensory_signals,ChemicalBathSchedule)
    from r_elegans.body import physical_muscle_body_params
    from r_elegans.model import load_builtin_model
    import jax.numpy as jnp
    with jax.experimental.enable_x64(True):
        sc=EmbodiedSensoryConfig(b_bend_gain_nS=0,dva_gain_nS=0,smdd_gain_nS=0)
        model=load_builtin_model();maps=build_sensory_maps(model,sc);params=physical_muscle_body_params(12,medium='liquid')
        times=np.arange(0,1.01,.01);events=np.array([0.,.2,.4,.6,.8]);levels=np.array([[0,0,50000],[0,0,60000],[0,0,50000],[1,0,50000],[0,0,50000.]])
        world=ChemicalBathSchedule(jnp.asarray(events),jnp.asarray(levels))
        body=SimpleNamespace(joint_angles=jnp.zeros(11),position=jnp.zeros(2),heading=jnp.asarray(0.),time=jnp.asarray(0.))
        state=initial_embodied_sensory_state(body,params,world,sc)
        replay=replay_chemical(times,events,levels,state.receptor,state.adapted,np.asarray(sc.chemical_reference_um),sc.receptor_time_constant_s,sc.adaptation_time_constant_s,sc.chemical_gain_nS)
        expected=np.zeros_like(replay);clock=0.
        for i in range(1,len(times)):
            body.time=jnp.asarray(clock)
            state,_,observation=transduce_sensory_signals(state,body,params,world,maps,.01,sc)
            expected[i]=np.asarray(observation.chemical_excitatory_nS)[np.asarray(maps.chemical_indices)]
            clock+=.01
        np.testing.assert_allclose(replay,expected,rtol=1e-10,atol=1e-14)
        assert replay[:,4].max()>0 and replay[:,5].max()>0 and replay[:,0].max()>0
    with pytest.raises(ValueError):replay_chemical([0],events,levels,[0]*3,[0]*3,np.ones(3),1,1,1)
