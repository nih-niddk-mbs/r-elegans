"""E048 fixed-winner world/body -> sensory cells -> recurrent brain diagnostic.

Chemical and proprioceptive blocks are causal controls. Wet-agar body mechanics
are interleaved step by step with the brain, never replayed after recording it.
No optimization, periodic gait teacher, output clamp or checkpoint promotion.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import shutil
import time

from scripts.assess_wet_surface import verify_reference, WINNER_SHA256, READOUT_SHA256
from scripts.experiment_records import capture_run_record, file_sha256


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_state(path, state, reference_velocity=None):
    import numpy as np
    values = {'brain_'+k:np.asarray(v) for k,v in state.brain_body.brain._asdict().items()}
    values.update({'body_'+k:np.asarray(v) for k,v in state.brain_body.body._asdict().items()})
    values.update({'sensory_'+k:np.asarray(v) for k,v in state.sensory._asdict().items()})
    if reference_velocity is not None:
        values['reference_velocity'] = np.asarray(reference_velocity)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def arrays_from_trace(trace):
    import numpy as np
    values = {k:np.asarray(getattr(trace, k)) for k in ('voltage_mV','calcium_uM','muscles','input_current_pA')}
    values.update({'body_'+k:np.asarray(v) for k,v in trace.body._asdict().items()})
    values.update({'sensory_'+k:np.asarray(v) for k,v in trace.sensory._asdict().items()})
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=8.)
    parser.add_argument('--dt', type=float, default=.002)
    parser.add_argument('--max-brain-step-ms', type=float, default=.1)
    parser.add_argument('--medium', choices=('liquid','wet-agar','both'), default='both')
    parser.add_argument('--refine', action='store_true', help='Include half-body-timestep closed-loop controls')
    args = parser.parse_args()
    if (not all(math.isfinite(x) and x>0 for x in (args.duration,args.dt,args.max_brain_step_ms))
            or args.duration < 5 or args.duration > 600 or args.dt > .002):
        parser.error('Duration 5..600s, positive dt<=.002s and positive brain step required')
    repo = Path(__file__).resolve().parents[1]
    source, network, readout = verify_reference(args.reference)
    out = args.output_dir.expanduser().resolve()
    if out == repo or repo in out.parents or out == source or source in out.parents:
        parser.error('Keep outputs outside repository and parent checkpoint')
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    capture_run_record(out, repo, args)
    inputs = out/'inputs'
    inputs.mkdir()
    for path, name in ((network,'network.npz'),(readout,'parent_actor.npz')):
        shutil.copy2(path, inputs/name)
    if file_sha256(inputs/'network.npz') != WINNER_SHA256 or file_sha256(inputs/'parent_actor.npz') != READOUT_SHA256:
        raise ValueError('Copied input checksum mismatch')

    import jax
    import jax.numpy as jnp
    import numpy as np
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.body import physical_muscle_body_params, build_muscle_projection
    from r_elegans.body.closed_loop import (
        initial_embodied_state, embodied_channel_body_chunk, embodied_neural_step, EmbodiedTrace,
    )
    from r_elegans.body.embodied_sensory import (
        EmbodiedSensoryConfig, default_chemical_world, validate_chemical_world, build_sensory_maps, CUE_NAMES,
    )
    from r_elegans.body.wet_surface import wet_agar_drag, surface_body_step
    from r_elegans.rl.full_network_actor import actor_from_arrays
    from r_elegans.model import load_builtin_model
    from r_elegans.full_body_assessment import regional_amplitudes, write_motion_preview

    params, fixed, network_config = load_network(inputs/'network.npz')
    if np.any(np.asarray(fixed.bias_pA) != 0):
        raise ValueError('Pinned zero-bias network required')
    with np.load(inputs/'parent_actor.npz', allow_pickle=False) as z:
        parent = actor_from_arrays(z)
    model = load_builtin_model()
    base_sensory = EmbodiedSensoryConfig()
    world = default_chemical_world()
    validate_chemical_world(world)
    maps = build_sensory_maps(model, base_sensory)
    write_json(out/'configuration.json', dict(experiment='E048', source=str(source),
        network_sha256=WINNER_SHA256, readout_sha256=READOUT_SHA256,
        network_config=asdict(network_config), sensory_config=asdict(base_sensory),
        world={k:np.asarray(v).tolist() for k,v in world._asdict().items()}, cue_names=CUE_NAMES,
        input_units='geometry meters,concentration micromolar,conductance nS,current pA',
        transduction_status='Exploratory effective closure; gains/time constants/receptive fields not fitted',
        world_status='Prescribed Gaussian test fields with bath onset/washout,not fluid transport or food model'))
    np.savez_compressed(out/'sensory_maps.npz', **{k:np.asarray(v) for k,v in maps._asdict().items()})
    projection = build_muscle_projection(11, head_at_end=True)
    signed_projection = np.asarray(projection.dorsal_weights)-np.asarray(projection.ventral_weights)
    law = wet_agar_drag()
    media = ('liquid','wet-agar') if args.medium == 'both' else (args.medium,)
    cases = [(medium, mode, args.dt) for medium in media
             for mode in ('closed-loop','chemical-blocked','proprioceptive-blocked','all-blocked')]
    if args.refine:
        cases += [(medium,'closed-loop-refined',args.dt/2) for medium in media]
    rows = []
    started = time.monotonic()
    for medium, mode, dt in cases:
        case = medium+'-'+mode
        destination = out/case
        destination.mkdir()
        sc = replace(base_sensory,
            chemical_gain_nS=0. if mode in ('chemical-blocked','all-blocked') else base_sensory.chemical_gain_nS,
            b_bend_gain_nS=0. if mode in ('proprioceptive-blocked','all-blocked') else base_sensory.b_bend_gain_nS,
            dva_gain_nS=0. if mode in ('proprioceptive-blocked','all-blocked') else base_sensory.dva_gain_nS)
        body = physical_muscle_body_params(12, medium='liquid' if medium == 'liquid' else 'agar')
        state = initial_embodied_state(fixed, network_config, body, world, sc)
        if medium == 'wet-agar':
            state = state._replace(brain_body=state.brain_body._replace(
                body=jax.tree_util.tree_map(lambda x:np.asarray(x,dtype=float), state.brain_body.body)))
        substeps = math.ceil(dt*1000/args.max_brain_step_ms)
        chunk_steps = round(1./dt)
        if medium == 'liquid':
            advance = jax.jit(lambda s:embodied_channel_body_chunk(s, params, fixed, network_config,
                parent, model, body, world, maps, dt=dt, steps=chunk_steps,
                brain_substeps=substeps, sensory_config=sc))
        else:
            neural_step = jax.jit(lambda s:embodied_neural_step(s, params, fixed, network_config,
                parent, model, body, world, maps, dt, substeps, sc))
        save_state(destination/'state_initial.npz', state)
        write_json(destination/'parameters.json', dict(sensory=asdict(sc),
            body={k:({a:float(b) for a,b in v._asdict().items()} if k=='mechanics' else float(v)) for k,v in body._asdict().items()},
            nonlinear_drag=asdict(law) if medium == 'wet-agar' else None,
            drag_note='Wet reference uses nonlinear law,not legacy linear fields' if medium == 'wet-agar' else 'Lighthill liquid'))
        buffers = []
        reference_velocity = None
        residual_max = 0.
        max_neural_change = 0.
        print(json.dumps(dict(starting=case, dt_s=dt, brain_substeps=substeps)), flush=True)
        for chunk in range(math.ceil(args.duration/(chunk_steps*dt))):
            if medium == 'liquid':
                state, trace = advance(state)
                arrays = arrays_from_trace(jax.device_get(trace))
            else:
                traces = []
                for _ in range(chunk_steps):
                    brain, memory, muscles, observed, current = neural_step(state)
                    command = signed_projection.astype(float) @ np.asarray(muscles, dtype=float)
                    next_body, reference_velocity, info = surface_body_step(state.brain_body.body,
                        command, body, law, dt, initial_velocity=reference_velocity)
                    residual_max = max(residual_max, info['relative_force_residual'])
                    state = state._replace(brain_body=state.brain_body._replace(brain=brain, body=next_body), sensory=memory)
                    traces.append(arrays_from_trace(EmbodiedTrace(brain.voltage_mV, brain.calcium_uM,
                        muscles, next_body, observed, current)))
                arrays = {k:np.stack([row[k] for row in traces]) for k in traces[0]}
            if not all(np.isfinite(v).all() for v in arrays.values()) or not all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(state)):
                raise RuntimeError('Nonfinite feedback state in '+case)
            file = destination/f'chunk-{chunk:04d}.npz'
            np.savez_compressed(file, **arrays)
            buffers.append(arrays)
            save_state(destination/'state_latest.npz', state, reference_velocity)
            print(json.dumps(dict(case=case, simulation_seconds=(chunk+1)*chunk_steps*dt,
                elapsed_seconds=time.monotonic()-started)), flush=True)
        combined = {k:np.concatenate([b[k] for b in buffers]) for k in buffers[0]}
        np.savez_compressed(destination/'trajectory.npz', **combined, dt_s=dt)
        save_state(destination/'state_final.npz', state, reference_velocity)
        regions = regional_amplitudes(combined['body_joint_angles'], combined['body_muscle_activation'])
        write_motion_preview(destination/'motion.html', combined['body_joint_angles'], combined['body_position'],
            combined['body_heading'], dt=dt, body_length=.001, seconds=5., title='E048 sensory-connected / '+case)
        row = dict(case=case, medium=medium, mode=mode, dt_s=dt, brain_substeps=substeps,
            simulated_seconds=len(combined['voltage_mV'])*dt,
            body_feedback=sc.b_bend_gain_nS>0 or sc.dva_gain_nS>0,
            chemical_input=sc.chemical_gain_nS>0, regions=regions,
            maximum_input_current_pA=float(np.max(np.abs(combined['input_current_pA']))),
            maximum_chemical_conductance_nS=float(np.max(combined['sensory_chemical_excitatory_nS']+combined['sensory_chemical_inhibitory_nS'])),
            maximum_proprioceptive_conductance_nS=float(combined['sensory_proprioceptive_nS'].max()),
            maximum_wet_force_residual=residual_max if medium == 'wet-agar' else None,
            parameters_trained=False, locomotion_qualified=False)
        write_json(destination/'summary.json', row)
        rows.append(row)
        write_json(out/'progress.json', dict(completed=[r['case'] for r in rows],elapsed_seconds=time.monotonic()-started))
    comparisons = []
    refinements = []
    for medium in media:
        with np.load(out/(medium+'-closed-loop')/'trajectory.npz') as z:
            baseline = {k:z[k] for k in ('voltage_mV','muscles','body_joint_angles')}
        for mode in ('chemical-blocked','proprioceptive-blocked','all-blocked'):
            with np.load(out/(medium+'-'+mode)/'trajectory.npz') as z:
                comparisons.append(dict(medium=medium, compared_to=mode,
                    max_voltage_difference_mV=float(np.max(np.abs(baseline['voltage_mV']-z['voltage_mV']))),
                    max_muscle_difference=float(np.max(np.abs(baseline['muscles']-z['muscles']))),
                    max_joint_difference_rad=float(np.max(np.abs(baseline['body_joint_angles']-z['body_joint_angles']))),
                    maximum_control_voltage_difference_mV={name:float(np.max(np.abs(baseline['voltage_mV'][:,model.neuron_ids.index(name)]-z['voltage_mV'][:,model.neuron_ids.index(name)])))
                        for name in ('AVBL','AVBR','AVAL','AVAR','RIS')}))
        if args.refine:
            with np.load(out/(medium+'-closed-loop-refined')/'trajectory.npz') as z:
                # Record end-of-interval samples: every second fine sample
                # aligns with the coarse endpoint. Source pulses are sampled by
                # each physical integration, so this includes event discretization.
                fine_q = z['body_joint_angles'][1::2]
                fine_v = z['voltage_mV'][1::2]
                refinements.append(dict(medium=medium,
                    max_joint_difference_rad=float(np.max(np.abs(baseline['body_joint_angles']-fine_q))),
                    max_voltage_difference_mV=float(np.max(np.abs(baseline['voltage_mV']-fine_v))),
                    interpretation='Descriptive convergence check including stimulus-event sampling,not a biological pass'))
    if file_sha256(network) != WINNER_SHA256 or file_sha256(readout) != READOUT_SHA256:
        raise ValueError('Source checkpoint changed during diagnostic')
    summary = dict(experiment='E048',status='completed',cases=rows,paired_comparisons=comparisons,
        timestep_comparisons=refinements,
        elapsed_seconds=time.monotonic()-started,network_sha256=WINNER_SHA256,readout_sha256=READOUT_SHA256,
        parameters_trained=False,teacher_or_pacing=False,biological_sensing_validated=False,
        interpretation='Connections and causal effects tested;not food finding,regulated start-stop or validated locomotion',
        limitations=['Transduction constants,adaptive proxy,receptive fields and DVA bending proxy are exploratory.',
            'No exact receptor cascades,contact/water sensor or monoamine/neuropeptide mechanism.',
            'Gaussian test plumes are prescribed;not dish-scale chemical transport or nutrient uptake.',
            'Wet mechanics retains E046 low-speed-law uncertainties;no spatial convergence claimed.',
            'Modulated control-neuron voltage does not establish its causal control role.'])
    write_json(out/'summary.json', summary)
    manifest = {str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'artifact_manifest.json', manifest)
    print('EMBODIED_SENSES_EXIT_CODE=0',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
