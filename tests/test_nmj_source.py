import numpy as np
import pytest
from r_elegans.motor_identity import is_motor,output_group
from r_elegans.nmj_source import release_hold


def test_motor_identity_excludes_every_as_sensory_cell():
    for n in ('ASGL','ASGR','ASHL','ASHR','ASIL','ASIR','ASJL','ASJR','ASKL','ASKR','ASEL','ASER'):
        assert not is_motor(n) and output_group(n)=='other'
    for n in ('AS01','AS11','DB01','VB11','DA01','RMDL','SMDVR','SMBDL'):
        assert is_motor(n)


def test_corrected_group_schema_excludes_as_sensory_channels():
    from test_cma_curriculum import grouped_fixture
    from r_elegans.avb_training import parameter_groups
    spec,ids,origin,lower,upper,_=grouped_fixture()
    groups=parameter_groups(spec,ids,origin,lower,upper);ci=np.nonzero(spec.channels)
    AS=[g for g in groups if g['name'].startswith('density:AS:')]
    assert len(AS)==2
    for g in AS:
        assert {ids[i] for i in ci[0][g['indices']]}=={f'AS{i:02d}' for i in range(1,12)}


def test_nmj_hold_is_selective_reversible_and_preserves_nonzero_tone():
    import jax.numpy as jnp
    import jax
    ids=('DB01','VB01','RMDL','AS01','ASHL')
    threshold=jnp.full(5,-40.);slope=jnp.full(5,5.);mean=jnp.full(5,.3)
    voltage=jnp.array([-30.,-50.,-25.,-60.,-35.])
    live=jax.nn.sigmoid((voltage-threshold)/slope)
    hold=release_hold(ids,threshold,slope,mean,'B')
    for t in (0.,29.9,40.,59.):np.testing.assert_array_equal(hold(voltage,t),live)
    held=np.asarray(hold(voltage,35.));np.testing.assert_allclose(held[:2],.3)
    np.testing.assert_array_equal(held[2:],live[2:])
    with pytest.raises(ValueError):release_hold(ids,threshold,slope,np.ones(5)*2,'head')


def test_nmj_transfer_identity_and_intervention_have_no_parameter_mutation():
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from r_elegans.body.channel_bridge import channel_muscle_activations
    weights=jnp.ones((95,3));signs=jnp.ones((95,3))
    model=SimpleNamespace(neuromuscular_params=SimpleNamespace(synapse_weights=weights,synapse_signs=signs))
    parent=SimpleNamespace(raw_nmj=jnp.zeros((95,3)),unknown_polarity=jnp.zeros((95,3)),muscle_threshold=jnp.zeros(95),raw_muscle_slope=jnp.zeros(95))
    fixed=SimpleNamespace(release_threshold_mV=jnp.full(3,-40.),release_slope_mV=jnp.full(3,5.))
    voltage=jnp.array([-30.,-35.,-40.]);live=jax.nn.sigmoid((voltage+40)/5)
    normal=channel_muscle_activations(voltage,fixed,parent,model)
    np.testing.assert_array_equal(normal,channel_muscle_activations(voltage,fixed,parent,model,diagnostic_release=live))
    altered=channel_muscle_activations(voltage,fixed,parent,model,diagnostic_release=jnp.full(3,.1))
    assert np.all(np.asarray(altered)<np.asarray(normal))
    np.testing.assert_array_equal(channel_muscle_activations(voltage,fixed,parent,model),normal)
    with pytest.raises(ValueError):channel_muscle_activations(voltage,fixed,parent,model,diagnostic_release=jnp.zeros(2))


def test_closed_loop_keeps_recurrence_and_sensing_live_under_nmj_hold(monkeypatch):
    import jax.numpy as jnp
    from types import SimpleNamespace
    import r_elegans.body.closed_loop as loop
    from r_elegans.brain.conductance_network import MembraneConductanceInput
    old=SimpleNamespace(voltage_mV=jnp.array([-40.,-50.]));new=SimpleNamespace(voltage_mV=jnp.array([-35.,-45.]))
    state=SimpleNamespace(sensory='memory',brain_body=SimpleNamespace(brain=old,body=SimpleNamespace(time=35.)))
    membrane=MembraneConductanceInput(jnp.zeros(2),jnp.zeros(2));calls=[]
    monkeypatch.setattr(loop,'transduce_sensory_signals',lambda *a,**k:('updated',membrane,'observed'))
    def advance(*args,**kwargs):
        assert args[0] is old and kwargs['membrane_input'] is membrane
        calls.append('advanced');return new
    monkeypatch.setattr(loop,'advance_network',advance)
    def muscle(v,*args,**kwargs):
        np.testing.assert_array_equal(v,new.voltage_mV)
        return kwargs.get('diagnostic_release',jnp.ones(2))
    monkeypatch.setattr(loop,'channel_muscle_activations',muscle)
    ordinary=loop.embodied_neural_step(state,None,None,None,None,None,None,None,None,.002,20)
    held=loop.embodied_neural_step(state,None,None,None,None,None,None,None,None,.002,20,diagnostic_nmj_release=lambda v,t:jnp.full(2,.3))
    assert calls==['advanced','advanced'] and held[0] is new and held[1]=='updated'
    np.testing.assert_array_equal(ordinary[2],jnp.ones(2));np.testing.assert_allclose(held[2],.3)
