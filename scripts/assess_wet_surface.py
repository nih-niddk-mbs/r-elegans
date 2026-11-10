"""E046: measured wet-surface force law and unchanged autonomous-winner replay.

Generate endogenous commands once; compare bodies without fitting the neurons.
No feedback exists in this branch, so recording/replaying commands preserves its
one-way network/NMJ model. This is not a sensory or closed-loop crawling test.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.experiment_records import capture_run_record, file_sha256


WINNER_SHA256 = '417254ad65b18dd9b6f4ed5f4cf9502d6ad84f58e916d648471bccf16a417f14'
READOUT_SHA256 = '82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0'


def verify_reference(source):
    """Accept either immutable server worker or its E044 parameter reference."""
    source = Path(source).expanduser().resolve()
    if (source/'network_best.npz').is_file():
        from scripts.autonomous_warm_start import read_warm_start
        row = read_warm_start(source)
        summary = row['summary']
        network, readout = source/'network_best.npz', source/'source_inputs/parent_actor.npz'
    else:
        summary = json.loads((source/'worker_summary.json').read_text())
        manifest = json.loads((source/'reference_manifest.json').read_text())
        for row in manifest['files'].values():
            path = source/row['copy']
            if path.parent != source or file_sha256(path) != row['sha256']:
                raise ValueError('Invalid copied parameter-reference manifest')
        network, readout = source/'network.npz', source/'parent_actor.npz'
    if summary.get('passed') is not True or summary.get('status') != 'completed':
        raise ValueError('Completed qualified parent required')
    if file_sha256(network) != WINNER_SHA256 or file_sha256(readout) != READOUT_SHA256:
        raise ValueError('Expected pinned E043/E044 seed7001 and its frozen readout')
    return source, network, readout


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=20.)
    parser.add_argument('--brain-minutes', type=float, default=10.)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.duration, args.brain_minutes)) or args.duration < 20:
        parser.error('Duration >=20s and positive finite brain time budget required')
    repo = Path(__file__).resolve().parents[1]
    source, network, readout = verify_reference(args.reference)
    out = args.output_dir.expanduser().resolve()
    if out == repo or repo in out.parents or out == source or source in out.parents:
        parser.error('Keep results external and outside the parent')
    out.mkdir(parents=True, exist_ok=False)
    capture_run_record(out, repo, args)
    inputs = out/'inputs'
    inputs.mkdir()
    shutil.copy2(network, inputs/'network.npz')
    shutil.copy2(readout, inputs/'parent_actor.npz')
    if (file_sha256(inputs/'network.npz') != WINNER_SHA256 or
            file_sha256(inputs/'parent_actor.npz') != READOUT_SHA256):
        raise ValueError('Copied input checksum mismatch')
    for name in ('summary.json', 'worker_summary.json', 'reference_manifest.json', 'fit_policy.json',
                 'best.json', 'search_state.json', 'allowed_parameters.npz'):
        if (source/name).is_file():
            shutil.copy2(source/name, inputs/name)
    write_json(inputs/'manifest.json', dict(source=str(source),
        network_sha256=WINNER_SHA256, parent_actor_sha256=READOUT_SHA256,
        scope='Immutable evaluation inputs; not an optimizer continuation or complete historical backup'))

    # Preserve the existing network dtype/integration and avoid CUDA discovery on
    # CPU-only workers. The independent reference body uses NumPy float64.
    env = os.environ.copy()
    env.update(JAX_PLATFORMS='cpu', JAX_DEFAULT_MATMUL_PRECISION='highest')
    command = [sys.executable, '-u', '-m', 'scripts.diagnose_channel_body',
        '--network', str(inputs/'network.npz'), '--parent-actor', str(inputs/'parent_actor.npz'),
        '--output-dir', str(out/'endogenous-recording'), '--duration', str(args.duration),
        '--minutes', str(args.brain_minutes), '--medium', 'liquid', '--condition', 'active']
    print('Generating unchanged autonomous commands; log: '+str(out/'neural.log'), flush=True)
    with (out/'neural.log').open('w') as log:
        process = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=args.brain_minutes*60+90)
    if process.returncode:
        raise RuntimeError('Endogenous recording failed; see neural.log')
    recording = json.loads((out/'endogenous-recording/summary.json').read_text())
    if recording['stop_reason'] != 'duration' or recording['network_sha256'] != WINNER_SHA256:
        raise RuntimeError('Incomplete or mismatched autonomous recording; no surface comparison performed')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import numpy as np
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import load_network, initial_network_state
    from r_elegans.body import physical_muscle_body_params, build_muscle_projection
    from r_elegans.body.channel_bridge import channel_muscle_activations
    from r_elegans.body.wet_surface import wet_agar_drag, linear_drag, drag_terms, replay_surface_commands, SOURCE_DOI
    from r_elegans.full_body_assessment import regional_amplitudes, write_motion_preview
    from r_elegans.trajectory_analysis import analyze_trajectory, plot_analysis
    from r_elegans.rl.full_network_actor import actor_from_arrays
    from r_elegans.model import load_builtin_model

    params, fixed, config = load_network(inputs/'network.npz')
    if np.any(np.asarray(fixed.bias_pA) != 0):
        raise ValueError('Zero externally applied current required')
    with np.load(inputs/'parent_actor.npz', allow_pickle=False) as z:
        parent = actor_from_arrays(z)
    projection = build_muscle_projection(11, head_at_end=True)
    dorsal, ventral = np.asarray(projection.dorsal_weights), np.asarray(projection.ventral_weights)
    muscles, voltages, original_q = [], [], []
    chunks = []
    for name in recording['chunks']:
        path = out/'endogenous-recording'/name
        chunks.append(dict(file=name, sha256=file_sha256(path)))
        with np.load(path, allow_pickle=False) as z:
            muscles.append(z['muscles'])
            voltages.append(z['voltage_mV'])
            original_q.append(z['body_joint_angles'])
    muscles = np.concatenate(muscles)
    voltage = np.concatenate(voltages)
    original_q = np.concatenate(original_q)
    commands = muscles.astype(float) @ (dorsal-ventral).astype(float).T
    model = load_builtin_model()
    initial = initial_network_state(fixed, config)
    held_muscles = np.asarray(channel_muscle_activations(initial.voltage_mV, fixed, parent, model))
    held_commands = np.broadcast_to(held_muscles.astype(float) @ (dorsal-ventral).astype(float).T, commands.shape).copy()
    held_voltage = np.broadcast_to(np.asarray(initial.voltage_mV), voltage.shape)
    dt = recording['dt_s']
    np.savez_compressed(out/'endogenous_commands.npz', signed_commands=commands,
        held_voltage_commands=held_commands, dorsal_weights=dorsal, ventral_weights=ventral, dt_s=dt)
    body = physical_muscle_body_params(12, medium='agar')
    liquid = physical_muscle_body_params(12, medium='liquid')
    linear_liquid = linear_drag(float(liquid.mechanics.parallel_drag), float(liquid.mechanics.perpendicular_drag), name='bulk-water')
    linear_agar = linear_drag(float(body.mechanics.parallel_drag), float(body.mechanics.perpendicular_drag), name='legacy-agar')
    wet = wet_agar_drag()
    speed = np.geomspace(1e-8, 1e-3, 151)
    force_curves = {}
    for law in (linear_liquid, linear_agar, wet):
        force, _, _ = drag_terms(np.column_stack((speed, speed)), law)
        force_curves[law.name] = (force*.001).tolist()
    write_json(out/'force_reference.json', dict(source_doi=SOURCE_DOI,
        force_units='N for a 1mm contact length', speed_units='m/s', speed=speed.tolist(),
        curves=force_curves, surface=asdict(wet), published_fit_not_raw_data=True,
        near_zero_regularization_is_unmeasured=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for axis, label in enumerate(('Tangential', 'Normal')):
        for name, forces in force_curves.items():
            axes[axis].loglog(speed*1e6, np.asarray(forces)[:, axis]*1e6, label=name)
        axes[axis].axvline(wet.regularization_speed_m_s*1e6, color='gray', linestyle=':')
        axes[axis].set(xlabel='Towing speed (micrometers/s)', ylabel='Force for 1mm worm (microN)', title=label)
        axes[axis].legend(fontsize=8)
    fig.suptitle('Published wet-agar fit and model closures; curves are not raw measurements')
    fig.savefig(out/'force_reference.png', dpi=150)
    plt.close(fig)
    cases = [('liquid-reference', linear_liquid, 1, 4, False),
             ('legacy-agar', linear_agar, 1, 4, False),
             ('wet-agar', wet, 1, 4, False),
             ('wet-body-refined', wet, 2, 4, False),
             ('wet-quadrature-refined', wet, 1, 8, False),
             ('wet-floor-low', wet_agar_drag(regularization_speed_m_s=1e-8), 1, 4, False),
             ('wet-floor-high', wet_agar_drag(regularization_speed_m_s=1e-6), 1, 4, False),
             ('wet-held-voltage', wet, 1, 4, True)]
    rows = []
    started = time.monotonic()
    body_record = {k: ({a:float(b) for a,b in v._asdict().items()} if k == 'mechanics' else float(v))
                   for k, v in body._asdict().items()}
    write_json(out/'body_parameters.json', dict(parameters=body_record, segments=12, head_at_end=True,
        units='SI', primary_assessment_target='1.7% wet NGM agar empirical closure'))
    for name, law, substeps, quadrature, held in cases:
        print(json.dumps(dict(starting=name, samples=len(commands), substeps=substeps)), flush=True)
        destination = out/name
        destination.mkdir()
        def progress(completed, total):
            value = dict(case=name, completed_samples=completed, total_samples=total,
                         simulation_seconds=completed*dt, elapsed_seconds=time.monotonic()-started)
            write_json(out/'case-progress.json', value)
            print(json.dumps(value), flush=True)
        trace = replay_surface_commands(held_commands if held else commands, body, law, dt,
                                        substeps=substeps, quadrature_order=quadrature, progress=progress)
        np.savez_compressed(destination/'trajectory.npz', **trace, dt_s=dt)
        regions = regional_amplitudes(trace['joint_angles'], trace['muscle_activation'])
        metrics, derived = analyze_trajectory(held_voltage if held else voltage, trace['joint_angles'],
            trace['muscle_activation'], trace['position'], trace['heading'], dt=dt,
            body_length=float(body.mechanics.segment_length)*12, neuron_ids=model.neuron_ids)
        plot_analysis(metrics, derived, destination/'trajectory.png', f'E046 unchanged7001 / {name}')
        write_motion_preview(destination/'motion.html', trace['joint_angles'], trace['position'],
            trace['heading'], dt=dt, body_length=float(body.mechanics.segment_length)*12,
            seconds=5., title=f'E046 unchanged7001 / {name} / endogenous-command replay')
        row = dict(case=name, law=asdict(law), substeps=substeps, quadrature_order=quadrature,
            held_voltage_control=held, regions=regions, trajectory_analysis=metrics,
            raw_late_regional_median_p2p_rad=[float(np.median(group)) for group in
                np.array_split(np.ptp(trace['joint_angles'][len(commands)//2:], axis=0), 3)],
            maximum_relative_force_residual=float(trace['relative_force_residual'].max()),
            maximum_newton_iterations=int(trace['newton_iterations'].max()),
            maximum_point_speed_m_s=float(trace['maximum_point_speed_m_s'].max()),
            late_fraction_velocity_components_below_regularization=float(trace['fraction_components_below_regularization'][len(commands)//2:].mean()))
        if name == 'liquid-reference':
            row['maximum_joint_difference_from_original_jax_rad'] = float(np.max(np.abs(trace['joint_angles']-original_q)))
        write_json(destination/'report.json', row)
        rows.append(row)
        write_json(out/'progress.json', dict(completed=[r['case'] for r in rows], elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(finished=name, bends_tail_middle_head=regions['regional_joint_median_p2p_rad'],
            force_residual=row['maximum_relative_force_residual'],
            below_regularization=row['late_fraction_velocity_components_below_regularization'])), flush=True)
    base_row = next(r for r in rows if r['case'] == 'wet-agar')
    base = np.asarray(base_row['regions']['regional_joint_median_p2p_rad'])
    raw_base = np.asarray(base_row['raw_late_regional_median_p2p_rad'])
    sensitivity = {}
    for row in rows:
        if row['case'] in ('wet-body-refined', 'wet-quadrature-refined', 'wet-floor-low', 'wet-floor-high'):
            values = np.asarray(row['regions']['regional_joint_median_p2p_rad'])
            sensitivity[row['case']] = dict(relative_amplitude_change_with_003rad_floor=(np.abs(values-base)/np.maximum(base, .003)).tolist(),
                absolute_amplitude_change_rad=(values-base).tolist(),
                raw_amplitude_relative_change=(np.abs(np.asarray(row['raw_late_regional_median_p2p_rad'])-raw_base)/np.maximum(raw_base, 1e-15)).tolist())
    if file_sha256(network) != WINNER_SHA256 or file_sha256(readout) != READOUT_SHA256:
        raise ValueError('Source inputs changed during assessment')
    report = dict(experiment='E046', status='completed', network_sha256=WINNER_SHA256,
        parent_actor_sha256=READOUT_SHA256, recording_seconds=len(commands)*dt, dt_s=dt,
        parameters_trained=False, external_drive=False, body_to_brain_feedback=False,
        method='Recorded endogenous output of unchanged one-way channel network; independent float64 body replay',
        source=str(source), source_chunks=chunks, cases=rows, sensitivity=sensitivity,
        calibration_scope='Published towing-force fit reproduced, not independent gait validation',
        biological_crawling_validated=False,
        replay_elapsed_seconds=time.monotonic()-started,
        limitations=['Near-zero force law extrapolated and regularized; sensitivity is explicit.',
            'No dynamic groove memory, resolved capillarity/contact, self-contact, taper or natural plant geometry.',
            'No spatial convergence or refined neural integration in this assessment.',
            'Frequency cannot adapt to substrate without body-to-brain feedback in this branch.',
            'A short fixed-winner comparison is not robust motor-skill qualification or RL.'])
    write_json(out/'summary.json', report)
    files = {str(p.relative_to(out)):file_sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'artifact_manifest.json', files)
    print('WET_SURFACE_ASSESSMENT_EXIT_CODE=0', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
