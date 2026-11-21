"""Write partial / graded versions of the data-based synapse polarity for a checkpoint (reversal field only).

Each variant changes only `synapse_reversal_mV` on edges with a determined sign; parameters and every other fixed field are
copied unchanged. Used to find which parts of the polarity break the gait. Nothing is trained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.brain.conductance_network import load_network, network_synapses, save_network
from r_elegans.brain.edge_polarity import load_edge_polarity, polarized_reversal

try:
    from .experiment_records import capture_run_record
except ImportError:
    from experiment_records import capture_run_record

DESCRIPTIONS = dict(
    alpha010='full polarity blended 10 % of the way from the old reversal', alpha025='blended 25 %',
    alpha050='blended 50 %', alpha075='blended 75 %',
    inhibitory_only='only determined-inhibitory edges set to -85 mV', excitatory_only='only determined-excitatory edges set to 0 mV',
    nt_receptor_only='only edges with NT+receptor evidence (both signs)', inhibitory_mild70='full polarity with inhibitory reversal -70 mV',
    heavy50='full polarity on the heaviest determined edges covering half of the determined weight',
    light50='full polarity on the remaining determined edges', contradicting_only='only edges whose old reversal class contradicts the data sign')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant_matrices(old, sign, source, chemical):
    """Name -> [post, pre] reversal matrix; all variants equal `old` wherever sign == 0."""
    old_matrix = np.broadcast_to(np.asarray(old, float)[None, :], sign.shape).copy() if np.ndim(old) == 1 else np.asarray(old, float).copy()
    full = polarized_reversal(old_matrix, sign)
    edges = np.asarray(chemical) > 0
    determined = edges & (sign != 0)

    def only(mask):
        matrix = old_matrix.copy()
        matrix[mask] = full[mask]
        return matrix

    flat = np.flatnonzero(determined)
    order = flat[np.argsort(-np.asarray(chemical).ravel()[flat])]
    cumulative = np.cumsum(np.asarray(chemical).ravel()[order])
    heavy = np.zeros(sign.size, bool)
    heavy[order[:int(np.searchsorted(cumulative, .5 * cumulative[-1])) + 1]] = True
    heavy = heavy.reshape(sign.shape)
    category = np.where(old_matrix > -40, 1, np.where(old_matrix < -60, -1, 0))
    variants = {f'alpha{int(a * 100):03d}': old_matrix + a * (full - old_matrix) for a in (.10, .25, .50, .75)}
    variants.update(
        inhibitory_only=only(determined & (sign < 0)), excitatory_only=only(determined & (sign > 0)),
        nt_receptor_only=only(determined & (source == 1)), inhibitory_mild70=polarized_reversal(old_matrix, sign, inhibitory_mV=-70.),
        heavy50=only(determined & heavy), light50=only(determined & ~heavy),
        contradicting_only=only(determined & (category != sign)))
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path)
    parser.add_argument('--run', type=Path, help='with --checkpoint-sha256: find network.npz by hash under this run')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.run:
        from scripts.run_atlas_baseline import find_checkpoint
        args.network = find_checkpoint(args.run, args.checkpoint_sha256)
    if not args.network:
        parser.error('--network or --run with --checkpoint-sha256 required')
    repo = Path(__file__).resolve().parents[1]
    out = args.output_dir.expanduser().resolve()
    if out == repo or repo in out.parents:
        parser.error('Keep artifacts external')
    out.mkdir(parents=True, exist_ok=False)
    capture_run_record(out, repo, args)
    params, fixed, config = load_network(args.network)
    to64 = lambda a: jnp.asarray(a, jnp.float64) if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a
    params, fixed = jax.tree_util.tree_map(to64, (params, fixed))
    chemical = np.asarray(network_synapses(params, fixed, config)[0])
    polarity = load_edge_polarity()
    sign, source = polarity['sign'], polarity['source']
    old = np.asarray(fixed.synapse_reversal_mV)
    before = np.broadcast_to(old[None, :], sign.shape) if old.ndim == 1 else old
    edges = chemical > 0
    manifest = dict(input_network=str(args.network), input_sha256=sha256(args.network),
                    polarity_source_sha256=polarity['metadata']['source_sha256'], variants={},
                    scope='Reversal field only; parameters and all other fixed fields unchanged; not retrained')
    for name, matrix in variant_matrices(old, sign, source, chemical).items():
        (out / name).mkdir()
        save_network(out / name / 'network.npz', params, fixed._replace(synapse_reversal_mV=jnp.asarray(matrix)), config)
        changed = edges & ~np.isclose(before, matrix)
        manifest['variants'][name] = dict(description=DESCRIPTIONS[name], sha256=sha256(out / name / 'network.npz'),
                                          edges_changed=int(changed.sum()), weight_fraction_changed=float(chemical[changed].sum() / chemical.sum()))
        print(json.dumps({name: manifest['variants'][name]}), flush=True)
    (out / 'variants.json').write_text(json.dumps(manifest, indent=1) + '\n')


if __name__ == '__main__':
    main()
