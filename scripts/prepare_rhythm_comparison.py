"""Freeze verified forward inputs for E065; never fit or alter source models."""
import argparse
import json
import os
from pathlib import Path
import shutil

from scripts.experiment_records import capture_run_record, file_sha256


def prepare(old_reference, new_run, expected_new, output):
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    import numpy as np
    jax.config.update('jax_enable_x64', False)
    from scripts.assess_wet_surface import verify_reference, READOUT_SHA256
    from scripts.train_sensorimotor import verified_parent
    from scripts.sensorimotor_resume import read_completed
    from scripts.sensorimotor_parameters import load_saved_parameters
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.brain.single_compartment import CHANNEL_INDEX
    from r_elegans.rhythm_comparison import assert_fixed_equal
    old_reference = Path(old_reference).resolve(); new_run = Path(new_run).resolve()
    _, old_network, actor = verify_reference(old_reference)
    if file_sha256(new_run / 'network_best.npz') != expected_new:
        raise ValueError('Pinned new checkpoint mismatch')
    cfg = json.loads((new_run / 'configuration.json').read_text())
    if cfg['stage'] != 'alternation' or cfg['arguments']['precision'] != 'float64':
        raise ValueError('Expected current float64 alternation run')
    initial, fixed, config, spec, *_ = verified_parent(new_run / 'parent_run')
    offsets, sensory, _, _, summary = read_completed(new_run, cfg, spec.size)
    current, _, _ = load_saved_parameters(new_run, initial, fixed, config, spec, offsets)
    old, old_fixed, old_config = load_network(old_network)
    with jax.experimental.enable_x64(True):
        _, new_fixed, new_config = load_network(new_run / 'network_best.npz')
    assert_fixed_equal(old_fixed, new_fixed, old_config, new_config)
    assert_fixed_equal(old_fixed, fixed, old_config, config)
    if not np.array_equal(np.asarray(old.log_channel_scale)[:, CHANNEL_INDEX['LEAK']],
                          np.asarray(current.log_channel_scale)[:, CHANNEL_INDEX['LEAK']]):
        raise ValueError('Fixed leak channel scale changed')
    if cfg['readout_sha256'] != READOUT_SHA256 or file_sha256(actor) != READOUT_SHA256:
        raise ValueError('Frozen NMJ readout differs')
    if cfg['arguments']['dt'] != .002 or cfg['arguments']['brain_step_ms'] != .1:
        raise ValueError('Expected current .002s/.1ms integration')
    sources = {'old.npz': old_network, 'new.npz': new_run / 'network_best.npz', 'actor.npz': actor,
               'new_configuration.json': new_run / 'configuration.json',
               'new_summary.json': new_run / 'summary.json',
               'sensory_parameters.json': new_run / 'sensory_parameters.json',
               'parameter_reference.npz': new_run / 'parameter_reference.npz',
               'parameter_origin.npz': new_run / 'parameter_origin.npz',
               'reference-cue.npz': new_run / 'active/active-trajectory.npz',
               'reference-sham.npz': new_run / 'active/sham-trajectory.npz'}
    before = {name: file_sha256(path) for name, path in sources.items()}
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    for name, path in sources.items():
        shutil.copy2(path, out / name)
        if file_sha256(out / name) != before[name]:
            raise ValueError('Copied forward input mismatch')
    if before != {name: file_sha256(path) for name, path in sources.items()}:
        raise ValueError('Source changed during preparation')
    manifest = dict(experiment='E065', sources={name: str(path) for name, path in sources.items()},
                    sha256=before, fixed_compatible=True, fixed_leak_compatible=True,
                    new_checkpoint_sha256=summary['checkpoint_sha256'],
                    gain_multipliers=np.exp(sensory).tolist(), source_files_unchanged=True,
                    scope='Self-contained forward inference inputs; original optimizer/recursive training lineage remains in source, not a resume capsule')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    capture_run_record(out, Path(__file__).resolve().parents[1], dict(old_reference=str(old_reference), new_run=str(new_run)))
    print('Fixed fields/leak/readout compatible; forward inputs copied and hash-verified.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-reference', type=Path, required=True)
    p.add_argument('--new-run', type=Path, required=True)
    p.add_argument('--expected-new', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]; out = args.output_dir.resolve()
    for source in (repo, args.old_reference.resolve(), args.new_run.resolve()):
        if out == source or source in out.parents or out in source.parents:
            p.error('Output must be external and separate from source inputs')
    prepare(args.old_reference, args.new_run, args.expected_new, out)


if __name__ == '__main__':
    main()
