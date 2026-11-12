"""Exact frozen hybrid setup reused by every E067 connection-only search."""
import json
from pathlib import Path

import numpy as np


def setup(inputs):
    import jax
    jax.config.update('jax_enable_x64', False)
    from scripts.compare_rhythm_worker import verify_inputs
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.brain.joint_autonomous_fit import JointFitSpec, apply_joint_offsets, frozen_joint_fields_unchanged
    from r_elegans.brain.paired_fitting import make_rollout
    from r_elegans.body import physical_muscle_body_params, build_muscle_projection
    from r_elegans.body.closed_loop import initial_embodied_state
    from r_elegans.body.embodied_sensory import EmbodiedSensoryConfig, build_sensory_maps
    from r_elegans.model import load_builtin_model
    from r_elegans.rl.full_network_actor import actor_from_arrays
    from r_elegans.sensorimotor_curriculum import ResponseProtocol, cue_world
    from r_elegans.rhythm_comparison import assert_fixed_equal, array_fingerprint
    from r_elegans.connection_amplitude import connection_coordinates, inherited_bounds
    root=Path(inputs); manifest=verify_inputs(root)
    anchor, fixed, config=load_network(root/'old.npz')
    with jax.experimental.enable_x64(True):
        base, bf, bc=load_network(root/'candidate.npz')
        assert_fixed_equal(promote_floating_values(fixed),bf,config,bc)
    model=load_builtin_model()
    with np.load(root/'actor.npz',allow_pickle=False) as z:actor=actor_from_arrays(z)
    cfg=json.loads((root/'new_configuration.json').read_text())
    sc=EmbodiedSensoryConfig(**cfg['base_sensory_config'])
    world=cue_world(ResponseProtocol(**cfg['protocol']),sham=True)
    body=physical_muscle_body_params(12,medium='agar')
    templates=(body,build_sensory_maps(model,sc),build_muscle_projection(11,head_at_end=True))
    state=initial_embodied_state(fixed,config,body,world,sc)
    jax.config.update('jax_enable_x64',True)
    anchor,base,fixed,actor,model,world,templates,state=promote_floating_values(
        (anchor,base,fixed,actor,model,world,templates,state))
    if not np.array_equal(np.asarray(base.log_channel_scale),np.asarray(anchor.log_channel_scale)):
        raise ValueError('Restored channel scales differ from pinned old parent')
    spec=JointFitSpec(np.zeros_like(np.asarray(fixed.cells.conductances_nS),dtype=bool),
                      np.asarray(fixed.chemical_mask,dtype=bool),np.asarray(fixed.gap_mask,dtype=bool))
    origin=connection_coordinates(anchor,base,spec)
    lower,upper,slack=inherited_bounds(origin)
    source_cfg=json.loads((root/'training_parent_configuration.json').read_text())
    fingerprint=array_fingerprint(jax.tree_util.tree_leaves(state))
    if fingerprint!=source_cfg['base_initial_state_sha256']:
        raise ValueError('Training initial state differs from verified hybrid')
    gains=np.asarray(manifest['gain_multipliers'],dtype=float)
    def decode(x):
        candidate=apply_joint_offsets(base,np.asarray(x)-origin,spec)
        if not frozen_joint_fields_unchanged(candidate,base,spec):
            raise ValueError('Frozen channels or inactive connections changed')
        return candidate
    def factory(dt=.002,brain_step=.1,perturbation=0.,held=False):
        initial=state
        if perturbation:
            brain=state.brain_body.brain
            initial=state._replace(brain_body=state.brain_body._replace(
                brain=brain._replace(voltage_mV=brain.voltage_mV+perturbation)))
        return make_rollout(fixed,config,actor,model,medium='wet-agar',dt=dt,brain_step_ms=brain_step,
                            burn_in=0.,sampling='point',record_dense=True,hold_voltage=held,
                            sensory_config=sc,chemical_world=world,rollout_templates=templates,
                            initial_state=initial,exact_step_clock=True)
    return dict(manifest=manifest,anchor=anchor,base=base,fixed=fixed,config=config,model=model,
                state=state,spec=spec,origin=origin,lower=lower,upper=upper,slack=slack,
                fingerprint=fingerprint,gains=gains,decode=decode,factory=factory)
