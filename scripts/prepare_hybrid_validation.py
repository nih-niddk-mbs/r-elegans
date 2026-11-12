"""Pin an E065 forward-only hybrid without changing its source checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shutil

from scripts.experiment_records import capture_run_record, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison-run', type=Path, required=True)
    p.add_argument('--expected-new', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--allow-short-reference', action='store_true', help='Explicit local software smoke only')
    args = p.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    jax.config.update('jax_enable_x64', False)
    import numpy as np
    from scripts.compare_rhythm_worker import verify_inputs
    from scripts.assess_wet_surface import WINNER_SHA256, READOUT_SHA256
    from scripts.assess_sensorimotor_resolution import promote_floating_values
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.rhythm_comparison import assert_fixed_equal, hybrid_parameters
    run = args.comparison_run.resolve(); out = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    for source in (repo, run):
        if out == source or source in out.parents or out in source.parents:
            p.error('External output separate from comparison source required')
    summary = json.loads((run / 'summary.json').read_text())
    if not summary['execution_valid'] or summary['duration_s'] < (12 if args.allow_short_reference else 60):
        raise ValueError('Completed valid E065 source of sufficient duration required')
    manifest = verify_inputs(run / 'inputs')
    if (manifest['sha256']['old.npz'] != WINNER_SHA256
            or manifest['sha256']['new.npz'] != args.expected_new
            or manifest['sha256']['actor.npz'] != READOUT_SHA256):
        raise ValueError('Expected pinned parent families/readout')
    cfg = json.loads((run / 'old-new-sham/configuration.json').read_text())
    result = json.loads((run / 'old-new-sham/summary.json').read_text())
    candidate = run / 'old-new-sham/evaluated_network.npz'
    sha = file_sha256(candidate)
    if (sha != cfg['evaluated_network_sha256'] or not result['execution_valid']
            or cfg['case']['channels'] != 'old' or cfg['case']['connections'] != 'new'
            or cfg['case']['cue'] or cfg['case']['precision'] != 'float64'
            or cfg['case']['perturbation_mV'] != 0 or not cfg['case']['proprioception']
            or cfg['case']['brain_factor'] != 1 or cfg['dt_s'] != .002 or cfg['brain_step_ms'] != .1):
        raise ValueError('Expected unperturbed old/new sham candidate')
    old, fixed, config = load_network(run / 'inputs/old.npz')
    with jax.experimental.enable_x64(True):
        new, _, _ = load_network(run / 'inputs/new.npz')
        actual, af, ac = load_network(candidate)
        expected = hybrid_parameters(promote_floating_values(old), new, 'old', 'new')
        assert_fixed_equal(promote_floating_values(fixed), af, config, ac)
        if any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(expected, actual)):
            raise ValueError('Candidate does not match exact channel/synapse family reconstruction')
    sources = {name: run / 'inputs' / name for name in manifest['sha256']}
    sources.update({'candidate.npz': candidate,
                    'candidate_configuration.json': run / 'old-new-sham/configuration.json',
                    'candidate_summary.json': run / 'old-new-sham/summary.json',
                    'candidate_initial_state.npz': run / 'old-new-sham/initial_state.npz',
                    'comparison_input_manifest.json': run / 'inputs/manifest.json',
                    'reference-hybrid.npz': run / 'old-new-sham/trajectory.npz',
                    'comparison_summary.json': run / 'summary.json'})
    hashes = {name: file_sha256(path) for name, path in sources.items()}
    out.mkdir(parents=True, exist_ok=False)
    for name, path in sources.items():
        shutil.copy2(path, out / name)
        if file_sha256(out / name) != hashes[name]:
            raise ValueError('Copied candidate input mismatch')
    if hashes != {name: file_sha256(path) for name, path in sources.items()}:
        raise ValueError('Source changed during snapshot')
    manifest.update(experiment='E066', protocol='hybrid', sha256=hashes,
                    sources={name: str(path) for name, path in sources.items()},
                    candidate_sha256=sha, candidate_case='old-new-sham', source_run=str(run),
                    source_reference_seconds=summary['duration_s'],
                    allow_short_reference=args.allow_short_reference, source_files_unchanged=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    capture_run_record(out, repo, args)
    print(json.dumps(dict(candidate_sha256=sha, exact_family_reconstruction=True, source_unchanged=True)), flush=True)


if __name__ == '__main__':
    main()
