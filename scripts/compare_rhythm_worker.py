"""One unpaced E065 forward simulation with a frozen parameter-family hybrid."""
import argparse
import json
import os
from pathlib import Path
import time

from scripts.experiment_records import capture_run_record, file_sha256


def verify_inputs(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, sha in manifest['sha256'].items():
        if Path(name).name != name or file_sha256(root / name) != sha:
            raise ValueError('Forward input hash mismatch: ' + name)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--duration', type=float, default=60.)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64', False)
    from r_elegans.rhythm_comparison import comparison_cases, hybrid_parameters, array_fingerprint
    from r_elegans.signal_chain_audit import FIELDS, summarize_trace, plot_raw_chain
    from r_elegans.brain.conductance_network import load_network, save_network
    from r_elegans.brain.paired_fitting import make_rollout
    from r_elegans.body import physical_muscle_body_params, build_muscle_projection
    from r_elegans.body.closed_loop import initial_embodied_state
    from r_elegans.body.embodied_sensory import EmbodiedSensoryConfig, build_sensory_maps
    from r_elegans.model import load_builtin_model
    from r_elegans.rl.full_network_actor import actor_from_arrays
    from r_elegans.sensorimotor_curriculum import ResponseProtocol, cue_world
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    from scripts.assess_wet_surface import write_json
    cases = {c['name']: c for c in comparison_cases()}
    if args.case not in cases or not np.isfinite(args.duration) or args.duration < 12:
        p.error('Known case and duration>=12s required')
    case = cases[args.case]; root = args.inputs.resolve(); out = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    for source in (repo, root):
        if out == source or source in out.parents or out in source.parents:
            p.error('Keep worker output external and separate from inputs')
    manifest = verify_inputs(root)
    cfg = json.loads((root / 'new_configuration.json').read_text())
    old, fixed, config = load_network(root / 'old.npz')
    with jax.experimental.enable_x64(True):
        new, _, _ = load_network(root / 'new.npz')
    model = load_builtin_model()
    with np.load(root / 'actor.npz', allow_pickle=False) as z:
        actor = actor_from_arrays(z)
    sc = EmbodiedSensoryConfig(**cfg['base_sensory_config'])
    protocol = ResponseProtocol(**cfg['protocol'])
    worlds = {name: cue_world(protocol, sham=sham) for name, sham in (('cue', False), ('sham', True))}
    body = physical_muscle_body_params(12, medium='agar')
    templates = (body, build_sensory_maps(model, sc), build_muscle_projection(11, head_at_end=True))
    state = initial_embodied_state(fixed, config, body, worlds['sham'], sc)
    if case['precision'] == 'float64':
        jax.config.update('jax_enable_x64', True)
        old, new, fixed, actor, model, templates, state, worlds = promote_floating_values(
            (old, new, fixed, actor, model, templates, state, worlds))
    else:
        # Float32 controls use only old/old; never downcast a fitted64 model.
        assert case['channels'] == case['connections'] == 'old'
    params = hybrid_parameters(old, new, case['channels'], case['connections'])
    base_fingerprint = array_fingerprint(jax.tree_util.tree_leaves(state))
    brain = state.brain_body.brain
    state = state._replace(brain_body=state.brain_body._replace(brain=brain._replace(
        voltage_mV=brain.voltage_mV + np.asarray(case['perturbation_mV'], dtype=brain.voltage_mV.dtype))))
    initial_fingerprint = array_fingerprint(jax.tree_util.tree_leaves(state))
    gains = np.asarray(manifest['gain_multipliers'], dtype=fixed.initial_voltage_mV.dtype)
    if not case['proprioception']:
        gains[1:] = 0.
    out.mkdir(parents=True, exist_ok=False)
    os.environ['JAX_ENABLE_X64'] = str(case['precision'] == 'float64').lower()
    capture_run_record(out, repo, args)
    save_network(out / 'evaluated_network.npz', params, fixed, config)
    leaves = jax.tree_util.tree_leaves(state)
    np.savez_compressed(out / 'initial_state.npz', **{f'leaf_{i:03d}': np.asarray(x) for i, x in enumerate(leaves)})
    settings = dict(experiment='E065', case=case, duration_s=args.duration,
                    source_input_manifest_sha256=file_sha256(root / 'manifest.json'),
                    evaluated_network_sha256=file_sha256(out / 'evaluated_network.npz'),
                    precision=case['precision'], base_initial_state_sha256=base_fingerprint,
                    initial_state_sha256=initial_fingerprint, initial_state_tree=str(jax.tree_util.tree_structure(state)),
                    gain_multipliers=gains.tolist(), dt_s=.002, brain_step_ms=.1 * case['brain_factor'],
                    exact_step_clock=True, no_optimization=True, no_rng_draws=True,
                    initialization='Legacy32 body/maps/gates/calcium/sensory state widened exactly for64; t0 voltage-only perturbation if specified')
    write_json(out / 'configuration.json', settings)
    run = make_rollout(fixed, config, actor, model, medium='wet-agar', dt=.002,
                       brain_step_ms=settings['brain_step_ms'], burn_in=0., sampling='point', record_dense=True,
                       sensory_config=sc, chemical_world=worlds['cue' if case['cue'] else 'sham'],
                       rollout_templates=templates, initial_state=state, exact_step_clock=True)
    started = time.monotonic()
    trace = run(params, np.linspace(0., args.duration, 121), gains)
    elapsed = time.monotonic() - started
    expected_dtype = np.float64 if case['precision'] == 'float64' else np.float32
    if trace['dense_voltage_mV'].dtype != expected_dtype:
        raise ValueError('Requested neural precision not preserved')
    np.savez_compressed(out / 'trajectory.npz', **trace)
    early = {k: trace[k][trace['dense_time_s'] <= 12 + 1e-12] for k in ('dense_time_s', *FIELDS)}
    early_stats = summarize_trace(early, model.neuron_ids, 4.)
    late_stats = summarize_trace(trace, model.neuron_ids, args.duration / 2.)
    write_json(out / 'early-channels.json', early_stats)
    write_json(out / 'late-channels.json', late_stats)
    plot_raw_chain(trace, late_stats, model.neuron_ids, out / 'raw-signals.png', args.case,
                   begin_s=args.duration / 2., show_cue=case['cue'])
    parity = None
    if args.case in ('new-new-cue', 'new-new-sham'):
        with np.load(root / ('reference-cue.npz' if case['cue'] else 'reference-sham.npz'), allow_pickle=False) as z:
            if not np.array_equal(z['dense_time_s'], early['dense_time_s']):
                raise ValueError('Saved early-reference time grid differs')
            errors = {k: float(np.max(np.abs(early[k] - z[k]))) for k in FIELDS}
            limits = {k: 5e-10 * (1 + float(np.max(np.abs(z[k])))) for k in FIELDS}
            parity = dict(errors=errors, tolerances=limits, passed=all(errors[k] <= limits[k] for k in FIELDS))
    verify_inputs(root)
    result = dict(experiment='E065', status='completed', case=case, elapsed_simulation_seconds=elapsed,
                  simulation_seconds=args.duration, initial_state_sha256=initial_fingerprint,
                  base_initial_state_sha256=base_fingerprint, input_hashes_unchanged=True,
                  early_reference_match=parity,
                  execution_valid=parity is None or parity['passed'],
                  late_repeating_by_stage={k: len(v['repeating_indices']) for k, v in late_stats.items()},
                  late_repeating_neurons=late_stats['dense_voltage_mV']['repeating_neurons'],
                  motor_named_groups=late_stats['dense_voltage_mV']['named_groups'],
                  scope='Finite-window descriptive repetitions; not a stable limit cycle, movement or biological qualification')
    write_json(out / 'summary.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['execution_valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
