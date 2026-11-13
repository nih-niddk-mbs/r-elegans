"""Audit saved E065/E066/E067 output paths without simulation or model changes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from r_elegans.output_path_audit import analyze, nmj_transfer
from scripts.experiment_records import capture_run_record, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison-run', type=Path, required=True)
    p.add_argument('--hybrid-run', type=Path)
    p.add_argument('--fit-run', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--software-smoke', action='store_true', help='Saved local short traces only; not server evidence')
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = a.output_dir.expanduser().resolve()
    roots = [r.expanduser().resolve() for r in (a.comparison_run, a.hybrid_run, a.fit_run) if r]
    if out == repo or repo in out.parents or any(out == r or out in r.parents or r in out.parents for r in roots):
        p.error('Separate external output directory required')
    tasks, sources = [], set()
    asset_path = repo/'r_elegans/assets/runtime_model_v1.json'
    sources.add(asset_path)
    asset = json.loads(asset_path.read_text())
    def add(label, root, case, network, actor, expected):
        trace = root/case/'trajectory.npz'
        if file_sha256(network) != expected:
            raise ValueError('Checkpoint hash mismatch: '+str(network))
        sources.update((trace, network, actor))
        tasks.append((label, trace, network, actor))
    comparison = roots[0]
    sources.add(comparison/'summary.json')
    if not json.loads((comparison/'summary.json').read_text())['execution_valid']:
        raise ValueError('Invalid comparison execution')
    for case in ('old-old-sham', 'old-new-sham'):
        cfgpath = comparison/case/'configuration.json'
        cfg = json.loads(cfgpath.read_text()); sources.add(cfgpath)
        add(case, comparison, case, comparison/case/'evaluated_network.npz',
            comparison/'inputs/actor.npz', cfg['evaluated_network_sha256'])
    if a.hybrid_run:
        root = a.hybrid_run.expanduser().resolve()
        s = json.loads((root/'summary.json').read_text()); sources.add(root/'summary.json')
        manifest = json.loads((root/'inputs/manifest.json').read_text()); sources.add(root/'inputs/manifest.json')
        if not s['execution_valid']:
            raise ValueError('Invalid source hybrid execution')
        for case in ('hybrid-sham', 'hybrid-long'):
            add(case, root, case, root/'inputs/candidate.npz', root/'inputs/actor.npz', manifest['candidate_sha256'])
    if a.fit_run:
        root = a.fit_run.expanduser().resolve()
        sources.add(root/'summary.json')
        cohort = json.loads((root/'summary.json').read_text())
        if len(cohort['workers']) != 4:
            raise ValueError('Four completed E067 workers required')
        for worker in cohort['workers']:
            seed = int(worker['seed']); folder = root/f'seed-{seed}'
            s = json.loads((folder/'summary.json').read_text()); sources.add(folder/'summary.json')
            if s['status'] != 'completed':
                raise ValueError('Worker has not completed')
            for case in ('ordinary', 'long'):
                add(f'seed-{seed}-{case}', folder, 'validation/'+case,
                    folder/'network_best.npz', root/'inputs/actor.npz', s['checkpoint_sha256'])
    before = {str(path): file_sha256(path) for path in sorted(sources)}
    out.mkdir(parents=True, exist_ok=False)
    capture_run_record(out, repo, a)
    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('input_hashes_before.json', before)
    rows = []
    actor_hashes = {file_sha256(task[3]) for task in tasks}
    if len(actor_hashes) != 1:
        raise ValueError('Compared traces must share the identical frozen muscle readout')
    if not a.software_smoke and actor_hashes != {'82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0'}:
        raise ValueError('Production audit requires the pinned historical muscle readout')
    for label, trace, network, actor in tasks:
        with np.load(trace, allow_pickle=False) as z, np.load(network, allow_pickle=False) as n, np.load(actor, allow_pickle=False) as b:
            duration = float(z['dense_time_s'][-1])
            if not a.software_smoke and duration < (300 if label.endswith('long') else 60)-1e-9:
                raise ValueError('Production audit requires full-duration source: '+label)
            window = duration/4 if label.endswith('long') or a.software_smoke else 15.
            drive, activation = nmj_transfer(z['dense_voltage_mV'], n, b, asset)
            transfer_error = float(np.max(np.abs(activation-z['dense_muscles'])))
            if transfer_error > 1e-7:
                raise ValueError(f'Frozen NMJ readout reconstruction differs: {label}: {transfer_error}')
            row = analyze(z, n['neuron_ids'].tolist(), b['raw_muscle_slope'], window_seconds=window, nmj_drive=drive)
            row['nmj_activation_reconstruction_error'] = transfer_error
        row.update(case=label, source_trace=str(trace), checkpoint_sha256=before[str(network)])
        save(label+'.json', row); rows.append(row)
        print(json.dumps(dict(case=label, cycles_tail_middle_head=row['minimum_joint_cycles_tail_middle_head'],
                              projection_error=row['projection_maximum_error'],
                              survival_ratio_last_window=row['windows'][-1]['balance']['survival_ratio'])), flush=True)
    after = {str(path): file_sha256(path) for path in sorted(sources)}
    save('input_hashes_after.json', after)
    if before != after or 'jax' in sys.modules:
        raise RuntimeError('Input changed or simulator imported during read-only audit')
    save('summary.json', dict(experiment='E068', execution_valid=True, source_files_unchanged=True,
        software_smoke=a.software_smoke, actor_sha256=next(iter(actor_hashes)), cases=rows,
        scope='Saved-trace description only; no simulation, fitting, new physical/biological qualification or changed acceptance gate',
        limitations=['D/V projection is verified within 1e-7 absolute, reflecting float32 map reconstruction.',
                     'Spectral survival describes a single observed motor frequency; null means no resolved carrier.',
                     'Sigmoid derivative is a local static output slope, not closed-loop causal gain.',
                     'Signed fan-in-normalized NMJ drive is algebraically reconstructed and verified against recorded activation; no per-edge current was recorded.',
                     'Long uses 150-225/225-300 s; ordinary uses 30-45/45-60 s. Different windows describe drift, not exact gains.',
                     'Analysis-time hashes do not reconstruct unrecorded historical environments.']))
    print('Audit complete: '+str(out/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
