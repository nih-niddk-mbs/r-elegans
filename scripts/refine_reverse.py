"""Time-bounded bidirectional refinement with original forward reference.

No RL. Full recurrent neural parameters learn, with fixed NMJs/muscle mapping.
Best actor is selected jointly on independent forward/reverse validation cases.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import actor_from_arrays, init_full_network_actor, make_full_network_actor_interface
from r_elegans.rl.motor_curriculum import (directional_muscle_sequence, motor_stimulation_vector,
    reverse_context_vector, traveling_bend_target, traveling_metrics)
from r_elegans.rl import bidirectional_curriculum, motor_curriculum, full_network_actor
from r_elegans.rl.bidirectional_curriculum import command_bank, validation_key
if __package__:
    from .train_timed import save_actor, save_json, checkpoint, finite_tree
else:
    from train_timed import save_actor, save_json, checkpoint, finite_tree


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward_reference(initial_actor):
    """Resolve the original forward teacher, never the degraded continuation."""
    manifest = json.loads((initial_actor.parent / "manifest.json").read_text())
    name = "forward_reference_actor" if "forward_reference_actor" in manifest else "initial_actor"
    digest = "forward_reference_sha256" if name == "forward_reference_actor" else "initial_actor_sha256"
    reference = Path(manifest[name]).expanduser().resolve()
    if not manifest.get(digest) or sha256(reference) != manifest[digest]:
        raise ValueError("Original forward reference missing or checksum mismatch")
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=480.)
    parser.add_argument("--iterations", type=int, default=1000000)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--forward-weight", type=float, default=4.)
    parser.add_argument("--validation-every", type=int, default=500)
    args = parser.parse_args()
    if args.steps < 128 or args.iterations < 1 or args.validation_every < 1:
        parser.error("Need >=128 steps and positive iterations/validation interval")
    if not all(math.isfinite(x) and x > 0 for x in (args.minutes, args.learning_rate, args.forward_weight)):
        parser.error("Minutes, learning rate and weight must be finite and positive")
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == repo or repo in output.parents:
        parser.error("Artifacts must stay outside Git")
    source = json.loads((args.initial_actor.parent / "manifest.json").read_text())
    if (source.get("reverse_context_scheme") != "fixed_random_NMJ_neurons_seed719_v1"
            or source.get("input_mode") != "current" or source.get("body_units") != "legacy_normalized"
            or source.get("muscle_projection") != "anterior_to_head_at_last_segment"):
        parser.error("Requires the fitted reverse checkpoint with matching input/body convention")
    try:
        reference_path = forward_reference(args.initial_actor)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    original_manifest = json.loads((reference_path.parent / "manifest.json").read_text())
    for field in ("amplitude", "stimulation_gain", "propagation_delay", "input_mode", "body_units", "muscle_projection"):
        if source.get(field) != original_manifest.get(field):
            parser.error(f"Original forward reference differs in {field}")
    gain, amplitude, delay, context_gain = (source[k] for k in
        ("stimulation_gain", "amplitude", "propagation_delay", "reverse_context_gain"))
    if (not all(math.isfinite(v) and v > 0 for v in (gain, amplitude, delay, context_gain))
            or amplitude > .1 or not .2 <= delay <= .8):
        parser.error("Invalid source motor parameters")
    model = load_builtin_model()
    expected = init_full_network_actor(model, jax.random.PRNGKey(0))
    def load(path):
        with np.load(path, allow_pickle=False) as data:
            a = actor_from_arrays(data)
        for name in ("chemical_mask", "gap_mask"):
            if not np.array_equal(getattr(a.neural, name), getattr(expected.neural, name)):
                parser.error("Checkpoint topology mismatch")
        for name in ("raw_nmj", "unknown_polarity", "muscle_threshold", "raw_muscle_slope"):
            if not np.array_equal(getattr(a, name), getattr(expected, name)):
                parser.error("This curriculum requires the original fixed NMJ/muscle mapping")
        return a
    actor, teacher = load(args.initial_actor), load(reference_path)
    output.mkdir(parents=True, exist_ok=False)
    started, stop = time.perf_counter(), False
    deadline = started + args.minutes*60
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    interface = make_full_network_actor_interface(model, train_group="neural", input_mode="current")
    stimulation, context = motor_stimulation_vector(model, gain), reverse_context_vector(model, context_gain)
    train = jnp.asarray(command_bank(args.steps))
    tuning = jnp.asarray(command_bank(args.steps*2, held_out=True))
    def predict(a, c, reverse):
        return directional_muscle_sequence(a, interface, c, stimulation, context=context, reverse=reverse)
    forward_train = jax.jit(jax.vmap(lambda c: predict(teacher, c, False)))(train)
    reverse_train = jax.vmap(lambda c: traveling_bend_target(c, amplitude=amplitude, delay=delay, reverse=True))(train)
    forward_tuning = jax.jit(jax.vmap(lambda c: predict(teacher, c, False)))(tuning)
    reverse_tuning = jax.vmap(lambda c: traveling_bend_target(c, amplitude=amplitude, delay=delay, reverse=True))(tuning)
    if not finite_tree((forward_train, reverse_train, forward_tuning, reverse_tuning)):
        raise RuntimeError("Nonfinite supervision")
    def loss(a, indices):
        c, f, r = train[indices], forward_train[indices], reverse_train[indices]
        forward = jax.vmap(lambda cmd: predict(a, cmd, False))(c)
        reverse = jax.vmap(lambda cmd: predict(a, cmd, True))(c)
        f_loss = jnp.mean((forward-f)**2) / jnp.maximum(jnp.mean((f-.4)**2), 1e-6)
        r_loss = jnp.mean((reverse-r)**2) / jnp.maximum(jnp.mean((r-.4)**2), 1e-6)
        return (args.forward_weight*f_loss+r_loss)/(args.forward_weight+1)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.learning_rate))
    opt_state = optimizer.init(actor)
    rng = jax.random.PRNGKey(args.seed)
    @jax.jit
    def update(a, state, key):
        key, batch_key = jax.random.split(key)
        indices = jax.random.choice(batch_key, len(train), (2,), replace=False)
        value, gradient = jax.value_and_grad(loss)(a, indices)
        updates, state = optimizer.update(gradient, state, a)
        return optax.apply_updates(a, updates), state, key, value
    @jax.jit
    def evaluate(a):
        return (jax.vmap(traveling_metrics)(jax.vmap(lambda c: predict(a,c,False))(tuning), forward_tuning),
                jax.vmap(traveling_metrics)(jax.vmap(lambda c: predict(a,c,True))(tuning), reverse_tuning))
    def validation(a):
        values = evaluate(a)
        if not finite_tree(values):
            raise RuntimeError("Nonfinite validation")
        f, r = ([{k: float(v[i]) for k,v in m.items()} for i in range(len(tuning))] for m in values)
        return dict(forward=f, reverse=r, key=validation_key(f,r))
    initial = validation(actor)
    best, best_validation, selected_iteration = actor, initial, 0
    iteration, status, error = 0, "completed", None
    save_actor(output / "actor_initial.npz", actor)
    save_actor(output / "actor_best.npz", best)
    save_json(output / "manifest.json", dict(stage="bidirectional_refinement", actor_neurons=302,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
        initial_actor=str(args.initial_actor), initial_actor_sha256=sha256(args.initial_actor),
        forward_reference_actor=str(reference_path), forward_reference_sha256=sha256(reference_path),
        amplitude=amplitude, propagation_delay=delay, stimulation_gain=gain,
        reverse_context_gain=context_gain, reverse_context_scheme=source["reverse_context_scheme"],
        input_mode="current", body_units="legacy_normalized", muscle_projection=source["muscle_projection"],
        train_group="neural", fixed_nmj=True, forward_weight=args.forward_weight,
        learning_rate=args.learning_rate, steps=args.steps, dt=.02, minutes=args.minutes,
        iterations=args.iterations, validation_every=args.validation_every,
        checkpoint_every=100, selection="joint_validation_failed_cases_then_worst_then_mean",
        training_periods=[2.1,2.7,3.3,4.1], tuning_periods=[2.3,3.1,3.9,4.5],
        train_cases=8, tuning_cases=8, batch_size=2, backend=jax.default_backend(), jax_version=jax.__version__,
        source_sha256={str(Path(p).resolve().relative_to(repo)): sha256(Path(p)) for p in
            (__file__, bidirectional_curriculum.__file__, motor_curriculum.__file__, full_network_actor.__file__)},
        git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True))))
    def consider(a, history):
        nonlocal best, best_validation, selected_iteration
        report = validation(a)
        history.write(json.dumps(dict(iteration=iteration, **report))+"\n")
        if tuple(report["key"]) < tuple(best_validation["key"]):
            best, best_validation, selected_iteration = a, report, iteration
            save_actor(output / "actor_best.npz", best)
        print(f"validation iteration={iteration} key={report['key']} selected_iteration={selected_iteration}",flush=True)
    try:
        with (output/"loss.jsonl").open("w",buffering=1) as losses, (output/"validation.jsonl").open("w",buffering=1) as history:
            history.write(json.dumps(dict(iteration=0, **initial))+"\n")
            while iteration < args.iterations and time.perf_counter() < deadline and not stop:
                candidate, state, key, value = jax.block_until_ready(update(actor,opt_state,rng))
                if not finite_tree((candidate,state,key,value)):
                    raise RuntimeError("Nonfinite update; retaining previous finite state")
                actor, opt_state, rng = candidate, state, key
                iteration += 1
                losses.write(json.dumps(dict(iteration=iteration,normalized_loss=float(value),elapsed_seconds=time.perf_counter()-started))+"\n")
                if iteration == 1 or iteration % 100 == 0:
                    checkpoint(output/"latest.msgpack",actor,opt_state,rng,iteration)
                    save_actor(output/"actor_latest.npz",actor)
                    print(f"iteration={iteration} normalized_loss={float(value):.6f} elapsed_min={(time.perf_counter()-started)/60:.1f}",flush=True)
                if iteration % args.validation_every == 0:
                    consider(actor,history)
            if iteration % args.validation_every:
                consider(actor,history)
        if stop:
            status = "interrupted"
        if iteration == 0:
            raise RuntimeError("Budget expired before first update")
    except Exception as exception:
        status, error = "failed", repr(exception)
    finally:
        checkpoint(output/"latest.msgpack",actor,opt_state,rng,iteration)
        save_actor(output/"actor_final.npz",actor)
        save_actor(output/"actor_best.npz",best)
    passed = best_validation["key"][0] == 0
    np.savez_compressed(output/"validation_commands.npz",training=np.asarray(train),tuning=np.asarray(tuning),
        stimulation=np.asarray(stimulation),reverse_context=np.asarray(context))
    result = dict(stage="bidirectional_refinement",status=status,error=error,passed=passed,
        iterations=iteration,selected_iteration=selected_iteration,initial=initial,final=best_validation,
        elapsed_seconds=time.perf_counter()-started,
        stop_reason="signal" if stop else "time_budget" if time.perf_counter()>=deadline else "iteration_cap",
        limitations="Selection uses imitation validation; both independent long physics assays remain required")
    save_json(output/"summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("initial","final")}),flush=True)
    return 0 if passed and status=="completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
