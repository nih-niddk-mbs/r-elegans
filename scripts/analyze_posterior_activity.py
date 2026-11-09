"""Locate weak posterior activity in saved validations; no fitting or simulation."""
import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from r_elegans.brain.body_amplitude import sustained_excursions
from r_elegans.brain.conductance_network import load_network
from r_elegans.body.channel_bridge import channel_muscle_activations
from r_elegans.body.neuromuscular import build_muscle_projection
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import actor_from_arrays
from scripts.experiment_records import capture_run_record, file_sha256


def analyze(worker):
    summary = json.loads((worker / 'summary.json').read_text())
    checkpoint = worker / 'network_best.npz'
    parent_path = worker / 'source_inputs/parent_actor.npz'
    trajectory = worker / 'validation-long.npz'
    _, fixed, _ = load_network(checkpoint)
    with np.load(parent_path, allow_pickle=False) as z:
        parent = actor_from_arrays(z)
    model = load_builtin_model()
    projection = build_muscle_projection(11, head_at_end=True)
    dorsal, ventral = map(np.asarray, projection)
    weights = np.asarray(model.neuromuscular_params.synapse_weights)
    with np.load(trajectory, allow_pickle=False) as z:
        voltage = z['voltage_mV']
        muscles = z['muscles']
        angles = z['body_joint_angles']
        filtered = z['body_muscle_activation']
    # Compare sampled saved muscles with the checkpoint transform before interpreting.
    sample = np.linspace(0, len(voltage) - 1, min(64, len(voltage)), dtype=int)
    rebuilt = np.asarray([channel_muscle_activations(jnp.asarray(voltage[i]), fixed, parent, model) for i in sample])
    error = float(np.max(np.abs(rebuilt - muscles[sample])))
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError(f'Saved muscle/readout mismatch: {error}')
    release = expit((voltage - np.asarray(fixed.release_threshold_mV)) / np.asarray(fixed.release_slope_mV))
    va, _ = sustained_excursions(voltage)
    ra, _ = sustained_excursions(release)
    ma, _ = sustained_excursions(muscles)
    d, v = muscles @ dorsal.T, muscles @ ventral.T
    da, _ = sustained_excursions(d)
    vnta, _ = sustained_excursions(v)
    neta, _ = sustained_excursions(d-v)
    fa, _ = sustained_excursions(filtered)
    qa, _ = sustained_excursions(angles)
    late = slice(len(voltage)//2, None)
    dptp, vptp, netptp = [np.ptp(x[late], axis=0) for x in (d, v, d-v)]
    rows = []
    for i in range(11):
        mask = (dorsal[i]+ventral[i]) > 0
        inputs = weights[mask].sum(axis=0) > 0
        rows.append(dict(joint=i, region='posterior' if i<4 else 'middle' if i<8 else 'anterior',
            input_neurons=int(inputs.sum()), rhythmic_input_neurons=int((va[inputs]>=.05).sum()),
            maximum_input_sustained_voltage_mV=float(va[inputs].max()),
            maximum_input_sustained_release=float(ra[inputs].max()),
            median_individual_muscle_sustained_p2p=float(np.median(ma[mask])),
            dorsal_sustained_p2p=float(da[i]), ventral_sustained_p2p=float(vnta[i]),
            net_sustained_p2p=float(neta[i]), filtered_sustained_p2p=float(fa[i]),
            joint_sustained_p2p_rad=float(qa[i]),
            late_net_over_sum_side_p2p=float(netptp[i]/(dptp[i]+vptp[i])) if dptp[i]+vptp[i]>0 else None))
    posterior = weights[(dorsal[:4]+ventral[:4]).sum(axis=0)>0].sum(axis=0)>0
    top = np.flatnonzero(posterior)[np.argsort(ra[posterior])[::-1][:12]]
    report = dict(worker=str(worker), sample_muscle_reconstruction_max_error=error,
        sha256={str(p.relative_to(worker)):file_sha256(p) for p in (worker/'summary.json',checkpoint,parent_path,trajectory)},
        parameter_coverage_by_kind=summary.get('parameter_coverage_by_kind'), joints=rows,
        strongest_posterior_release=[dict(neuron=model.neuron_ids[i], voltage_sustained_p2p_mV=float(va[i]),
            release_sustained_p2p=float(ra[i]), late_mean_release=float(release[late,i].mean())) for i in top])
    state_path, policy_path = worker/'search_state.json', worker/'fit_policy.json'
    if state_path.exists() and policy_path.exists():
        state = json.loads(state_path.read_text())
        policy = json.loads(policy_path.read_text())
        factor = policy.get('maximum_density_factor')
        if factor is not None and float(factor)>1:
            offsets = np.asarray(state['best'], dtype=float)
            bound = np.log(float(factor))
            report['search_bounds'] = dict(maximum_factor=float(factor),
                parameters_at_bound=int((np.abs(offsets)>=bound-1e-6).sum()),
                total_parameters=len(offsets))
        report['sha256'].update({str(p.relative_to(worker)):file_sha256(p) for p in (state_path,policy_path)})
    compact = dict(worker=worker.name, search_bounds=report.get('search_bounds'), regions=[])
    for region in ('posterior','middle','anterior'):
        group = [r for r in rows if r['region']==region]
        compact['regions'].append(dict(region=region, **{key:float(np.median([r[key] for r in group])) for key in (
            'rhythmic_input_neurons','maximum_input_sustained_voltage_mV','maximum_input_sustained_release',
            'median_individual_muscle_sustained_p2p','net_sustained_p2p','filtered_sustained_p2p','joint_sustained_p2p_rad')}))
    print(json.dumps(compact, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.expanduser().resolve(), args.output_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if out==repo or repo in out.parents or out==run or run in out.parents:
        parser.error('Keep output outside repository and source run')
    source = json.loads((run/'summary.json').read_text())
    workers = [run/f"seed-{w['seed']}" for w in source['workers']] if 'workers' in source else [run]
    out.mkdir(parents=True, exist_ok=False)
    capture_run_record(out, repo, args)
    report = dict(source_run=str(run), source_summary_sha256=file_sha256(run/'summary.json'),
        interpretation='Late-half repeated amplitudes, no fitting or simulation. Joint0 is tail.',
        limitations=['D/V cancellation ratio uses peak-to-peak ranges, not a causal decomposition.',
                     'Nonzero voltage does not imply effective synaptic release; both are reported.',
                     'No conclusion about locomotion or natural behavior follows.'], workers=[])
    failed = False
    for worker in workers:
        try:
            report['workers'].append(analyze(worker))
        except (OSError, ValueError, KeyError) as e:
            report['workers'].append(dict(worker=str(worker),error=str(e)))
            print(json.dumps(dict(worker=worker.name,error=str(e))), flush=True)
            failed = True
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    raise SystemExit(1 if failed else 0)


if __name__=='__main__':
    main()
