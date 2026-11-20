"""Write a copy of a conductance-network checkpoint whose synapse reversals follow measured/predicted edge polarity.

Edges with a data-supported sign (Fenyves et al. 2020: NT+receptor, then transmitter-only prediction) get the
excitatory or inhibitory reversal. Undetermined edges (complex, unpredicted, absent from the tables) keep the
checkpoint's existing presynaptic reversal ('keep', default) or a neutral value ('neutral'); they are listed, never
silently treated as known. Only the fixed reversal field changes; channel, edge and release parameters are copied.
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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path)
    parser.add_argument('--run', type=Path, help='with --checkpoint-sha256: find network.npz by hash under this run')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--excitatory-mV', type=float, default=0.)
    parser.add_argument('--inhibitory-mV', type=float, default=-85.)
    parser.add_argument('--undetermined', choices=('keep', 'neutral'), default='keep')
    parser.add_argument('--neutral-mV', type=float, default=-40.)
    args = parser.parse_args()
    if args.run:
        from scripts.run_atlas_baseline import find_checkpoint
        args.network = find_checkpoint(args.run, args.checkpoint_sha256)
    params, fixed, config = load_network(args.network)
    to64 = lambda a: jnp.asarray(a, jnp.float64) if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a
    params, fixed = jax.tree_util.tree_map(to64, (params, fixed))
    chemical = np.asarray(network_synapses(params, fixed, config)[0])
    polarity = load_edge_polarity()
    sign, source, conflict, ids = polarity['sign'], polarity['source'], polarity['conflict'], list(polarity['neuron_ids'])
    unknown = polarity['metadata']['table_names_unknown_to_model']
    old = np.asarray(fixed.synapse_reversal_mV)
    matrix = polarized_reversal(old, sign, excitatory_mV=args.excitatory_mV, inhibitory_mV=args.inhibitory_mV,
                                undetermined=args.undetermined, neutral_mV=args.neutral_mV)
    edges = chemical > 0
    before = np.broadcast_to(old[None, :], matrix.shape) if old.ndim == 1 else old
    changed = edges & ~np.isclose(before, matrix)
    out = args.output_dir.expanduser()
    out.mkdir(parents=True, exist_ok=False)
    save_network(out / 'network.npz', params, fixed._replace(synapse_reversal_mV=jnp.asarray(matrix)), config)
    np.savez_compressed(out / 'edge_polarity.npz', sign=sign, source=source, conflict=conflict, neuron_ids=np.array(ids))
    weight = chemical.sum()
    summary = dict(input_network=str(args.network), input_sha256=sha256(args.network), output_sha256=sha256(out / 'network.npz'),
                   polarity_source_sha256=polarity['metadata']['source_sha256'],
                   policy=dict(excitatory_mV=args.excitatory_mV, inhibitory_mV=args.inhibitory_mV, undetermined=args.undetermined,
                               neutral_mV=args.neutral_mV, precedence='NT+receptor, then transmitter-only'),
                   model_edges=int(edges.sum()), edges_nt_receptor=int((edges & (source == 1)).sum()), edges_nt_only=int((edges & (source == 2)).sum()),
                   edges_undetermined=int((edges & (sign == 0)).sum()), weight_fraction_determined=float(chemical[edges & (sign != 0)].sum() / weight),
                   sign_conflicts_between_methods=int((edges & conflict).sum()), edges_whose_reversal_changed=int(changed.sum()),
                   weight_fraction_changed=float(chemical[changed].sum() / weight),
                   determined_excitatory=int((edges & (sign > 0)).sum()), determined_inhibitory=int((edges & (sign < 0)).sum()),
                   names_unknown_to_model=unknown,
                   scope='Fixed reversal field only; the checkpoint was not retrained, so behavior and requalification are not established.')
    (out / 'polarity_provenance.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
