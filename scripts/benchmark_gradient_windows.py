"""Benchmark batched-window gradient throughput (CPU or GPU, optionally sharded across devices).

Measurement only: no parameters are updated and nothing is recorded as a result.
"""
import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from r_elegans.brain.conductance_network import load_network, network_recording_intervals
from r_elegans.brain.fit_constraints import conservative_fit_constraints, project_unknowns
from r_elegans.data.calcium import load_identified_calcium
from r_elegans.model import load_builtin_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--uid', required=True)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8, 32, 64])
    parser.add_argument('--max-step-ms', type=float, default=1.)
    parser.add_argument('--shard', action='store_true', help='Split the window batch across all visible devices')
    args = parser.parse_args()
    model = load_builtin_model()
    params, fixed, config = load_network(args.network)
    windows = []
    for start in np.linspace(0, 900, max(args.batches)).astype(int):
        windows.append(load_identified_calcium(args.recording, args.labels, model.neuron_ids, args.uid,
                                               frames=args.frames, start_frame=int(start)))
    names = windows[0]['names']
    if any(w['names'] != names for w in windows):
        raise SystemExit('Windows must share the same identified neurons')
    masks, _ = conservative_fit_constraints(fixed, model.neuron_ids, names)
    indices = jnp.asarray(windows[0]['indices'])
    intervals_all = jnp.asarray(np.stack([w['intervals_s'] for w in windows]))
    target_all = jnp.asarray(np.stack([w['normalized'] for w in windows]))
    substeps = math.ceil(float(intervals_all.max()) * 1000. / args.max_step_ms)
    devices = jax.devices()
    print(json.dumps(dict(devices=[str(d) for d in devices], x64=bool(jax.config.jax_enable_x64),
                          frames=args.frames, substeps_per_frame=substeps, observed=len(names))), flush=True)

    def window_prediction(p, intervals):
        return network_recording_intervals(project_unknowns(p, params, masks), fixed, intervals, substeps, config).indicator[:, indices]

    def batch_loss(p, intervals, target):
        prediction = jax.vmap(lambda i: window_prediction(p, i))(intervals)
        return jnp.mean((prediction - target) ** 2)

    step = jax.jit(jax.value_and_grad(batch_loss))
    for batch in args.batches:
        intervals, target = intervals_all[:batch], target_all[:batch]
        if args.shard and batch % len(devices) == 0 and len(devices) > 1:
            sharding = NamedSharding(Mesh(np.array(devices), ('b',)), P('b'))
            intervals, target = jax.device_put(intervals, sharding), jax.device_put(target, sharding)
        t0 = time.monotonic()
        value, grad = step(params, intervals, target)
        jax.block_until_ready(grad)
        compile_s = time.monotonic() - t0
        t0 = time.monotonic()
        for _ in range(3):
            value, grad = step(params, intervals, target)
        jax.block_until_ready(grad)
        run_s = (time.monotonic() - t0) / 3
        finite = all(bool(jnp.isfinite(x).all()) for x in jax.tree_util.tree_leaves(grad))
        print(json.dumps(dict(batch=batch, compile_plus_first_s=round(compile_s, 1), gradient_s=round(run_s, 3),
                              windows_per_s=round(batch / run_s, 2), loss=float(value), finite=finite,
                              sharded=bool(args.shard and len(devices) > 1 and batch % len(devices) == 0))), flush=True)


if __name__ == '__main__':
    main()
