"""Persistent single-GPU E113 evaluator; task files pin every model and state."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from scripts.experiment_records import capture_run_record, file_sha256
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.stable_atlas_policy import preserved

REPO = Path(__file__).resolve().parents[1]
FIELDS = ('voltage_mV', 'gates', 'calcium_uM', 'indicator')


def atomic_json(path, value):
    path = Path(path);temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n');temp.replace(path)


def evaluate(task, cache):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import load_network, ConductanceNetworkState
    from r_elegans.brain.common_state_replay import make_replay, collect_anchors
    from r_elegans.brain.selected_atlas_replay import rhythm_forward
    from r_elegans.brain.rhythm_qualification import rhythm_metrics, compare_cycles
    out = Path(task['output']);out.mkdir(parents=True, exist_ok=False)
    def pin(path, expected):
        if file_sha256(path) != expected:raise ValueError('Task input changed: '+str(path))
    pin(task['model'], task['model_sha256'])
    p, f, c = load_network(task['model'])
    p, f = jax.tree.map(lambda a: a.astype(jnp.float64) if jnp.issubdtype(a.dtype, jnp.floating) else a, (p, f))
    ids = saved_arrays(task['model'])['neuron_ids'].tolist();rmd = [ids.index(n) for n in ('RMDL', 'RMDR')]
    if task['kind'] == 'autonomy':
        starts, reference = collect_anchors(p, f, c, rmd, step_ms=.03125)
        anchor = out/'anchors.npz'
        np.savez_compressed(anchor, **{'start_'+k:np.asarray(getattr(starts, k)) for k in FIELDS})
        traces, metrics = [], []
        for step in (.03125, .015625):
            voltage = rhythm_forward(p, f, c, seconds=40., sample_dt=.002, max_step_ms=step, warmup_seconds=60.)
            traces.append(voltage[:, rmd]);metrics.append([rhythm_metrics(voltage[:, i], .002) for i in rmd])
            np.savez_compressed(out/('rhythm-'+str(step)+'.npz'), rmd_mV=traces[-1],
                                late_all_cell_std_mV=np.std(voltage[len(voltage)//2:], axis=0), sample_dt_s=.002)
        result = dict(anchors=str(anchor), anchors_sha256=file_sha256(anchor), rhythms=metrics,
            reference_indices=reference['indices'].tolist() if hasattr(reference['indices'], 'tolist') else reference['indices'],
            strict_cycle_checks=[compare_cycles(traces[0][:, j], traces[1][:, j], .002) for j in range(2)])
    else:
        pin(task['anchors'], task['anchors_sha256'])
        anchors = saved_arrays(task['anchors'])
        starts = ConductanceNetworkState(*(jnp.asarray(anchors['start_'+k], dtype=jnp.float64) for k in FIELDS))
        targets = task['targets'];step = task['step_ms']
        # Configuration is invariant across tasks; only params and phase states vary.
        config_key = json.dumps(saved_arrays(task['model'])['config_json'].tolist(), sort_keys=True)
        key = (step, config_key)
        if key not in cache:cache[key] = make_replay(c, rmd, step)
        forward = cache[key];means_all = np.empty((4, len(targets), len(ids)), dtype=np.float64)
        base = None;control_means = [];maximum_delta = 0.;labels = []
        for begin in range(0, len(targets), 8):
            chunk = targets[begin:begin+8];n = len(chunk);gs = np.zeros((10, len(ids)))
            gs[np.arange(n), [ids.index(t) for t in chunk]] = task['stimulus_nS']
            means, _, voltage = forward(p, f, starts, jnp.asarray(gs))
            means = np.asarray(means);means_all[:, begin:begin+n] = means[:, :n]
            zero = list(range(n, 10));v = np.asarray(voltage[:, jnp.asarray(zero)])
            if not np.isfinite(means).all() or not np.isfinite(v).all():raise ValueError('Nonfinite replay')
            if base is None:base = v[:, 0].copy()
            maximum_delta = max(maximum_delta, float(abs(v-base[:, None]).max()))
            control_means.append(np.moveaxis(means[:, zero], 1, 0))
            labels.extend([f'batch-{begin//8}-condition-{k}' for k in zero])
        cm = np.concatenate(control_means)
        response = means_all-cm[0][:, None]
        artifact = out/'responses.npz'
        np.savez_compressed(artifact, response=response, control_means=cm, reference_sham_rmd_mV=base,
            control_labels=labels, maximum_control_voltage_delta_mV=maximum_delta,
            targets=np.asarray(targets), neuron_ids=np.asarray(ids), anchors_sha256=task['anchors_sha256'], neural_step_ms=step)
        rhythms = [[rhythm_metrics(base[phase, :, j], .001) for j in range(2)] for phase in range(4)]
        result = dict(artifact=str(artifact), sha256=file_sha256(artifact),
            duplicate_mean_error=float(abs(cm-cm[0]).max()), duplicate_trace_error=maximum_delta,
            controls_passed=bool(abs(cm-cm[0]).max() <= 1e-12 and maximum_delta <= 1e-12), rhythms=rhythms)
        pin(task['anchors'], task['anchors_sha256'])
    pin(task['model'], task['model_sha256'])
    return result


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', type=Path, required=True);p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args();folder = args.root/'workers'/str(args.gpu);folder.mkdir(parents=True, exist_ok=True)
    try:
        import jax
        manifest = json.loads((args.root/'manifest.json').read_text());identity = manifest['forward_identity']
        if not jax.config.x64_enabled or jax.default_backend() != 'gpu' or len(jax.devices()) != 1:raise ValueError('One visible float64 GPU required')
        for name, sha in identity['forward_source_sha256'].items():
            if file_sha256(REPO/name) != sha:raise ValueError('Forward source differs: '+name)
        for name, version in identity['packages'].items():
            if importlib.metadata.version(name) != version:raise ValueError('Forward package differs: '+name)
        if sys.version != identity['python_version']:raise ValueError('Forward Python differs')
        for key, field in [('JAX_DEFAULT_MATMUL_PRECISION','matmul_setting'), ('XLA_FLAGS','xla_flags')]:
            if os.environ.get(key) != identity[field]:raise ValueError('Numerical environment differs: '+key)
        capture_run_record(folder, REPO, vars(args), phase='stable-atlas-persistent-worker')
        atomic_json(folder/'ready.json', dict(ready=True, gpu=args.gpu));cache = {}
        while not (folder/'STOP').exists():
            inbox = folder/'task.json'
            if not inbox.exists():time.sleep(.25);continue
            task = json.loads(inbox.read_text());inbox.unlink();start = time.monotonic()
            try:
                result = evaluate(task, cache)
                result.update(status='completed', task=task, elapsed_seconds=time.monotonic()-start)
            except Exception as exc:
                traceback.print_exc();result = dict(status='failed', error=repr(exc), task=task, elapsed_seconds=time.monotonic()-start)
            atomic_json(Path(task['result']), result)
    except Exception as exc:
        traceback.print_exc();atomic_json(folder/'startup-error.json', dict(error=repr(exc)));raise


if __name__ == '__main__':main()
