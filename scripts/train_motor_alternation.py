"""Supervised alternating/traveling bends; full recurrent brain, fixed NMJs.

Commands supply artificial motor-neuron currents, not food or biological sensory
input. Validation uses unseen command sequences and a longer horizon. No RL.
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
from r_elegans.rl.full_network_actor import (
    init_full_network_actor, actor_from_arrays, make_full_network_actor_interface,
)
from r_elegans.rl.motor_curriculum import (
    motor_stimulation_vector, commanded_bend_target, commanded_muscle_sequence,
    alternating_metrics, alternating_passes, muscle_sequence_body,
    traveling_bend_target, traveling_metrics, traveling_passes,
)
from r_elegans.rl import motor_curriculum, full_network_actor
if __package__:
    from .train_timed import finite_tree, save_actor, save_json, checkpoint
else:
    from train_timed import finite_tree, save_actor, save_json, checkpoint


def command_sequences(steps, *, held_out=False):
    """Slow scalar commands; training and validation differ in order/rate/level."""
    t = jnp.arange(steps) * 0.02
    if held_out:
        levels = jnp.asarray([0., -.75, .65, 0., .9, -.9, 0.])
        sine = .85 * jnp.sin(2 * jnp.pi * t / 3.7 - .7)
    else:
        levels = jnp.asarray([0., 1., -1., 0.])
        sine = jnp.sin(2 * jnp.pi * t / 2.4)
    index = jnp.minimum(jnp.arange(steps) * len(levels) // steps, len(levels) - 1)
    return jnp.stack([sine, levels[index]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-actor", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--stimulation-gain", type=float, default=0.1)
    parser.add_argument("--minutes", type=float, help="Optional wall-clock training budget, with checkpoint on exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=("alternation", "wave"), default="alternation")
    parser.add_argument("--propagation-delay", type=float, default=0.4)
    parser.add_argument("--amplitude", type=float, default=0.04, help="Synthetic muscle target amplitude; increase in small stages")
    args = parser.parse_args()
    if args.steps < 32 or args.iterations < 1:
        parser.error("At least 32 steps and one iteration required")
    if args.stage == "wave" and (args.steps < 128 or not .2 <= args.propagation_delay <= .8):
        parser.error("Wave fitting requires at least 128 steps and propagation delay in [0.2, 0.8] seconds")
    if not 0 < args.amplitude <= .1:
        parser.error("Amplitude must be in (0, 0.1]")
    for value in (args.learning_rate, args.stimulation_gain, args.propagation_delay, args.minutes if args.minutes is not None else 1.):
        if not math.isfinite(value) or value <= 0:
            parser.error("Rates, gain and optional minutes must be finite and positive")
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == repo or repo in output.parents:
        parser.error("Artifacts must stay outside Git")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + args.minutes * 60 if args.minutes else float("inf")
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    model = load_builtin_model()
    actor = init_full_network_actor(model, jax.random.PRNGKey(args.seed))
    if args.initial_actor:
        with np.load(args.initial_actor, allow_pickle=False) as data:
            restored = actor_from_arrays(data)
        for field in ("chemical_mask", "gap_mask"):
            if not np.array_equal(getattr(actor.neural, field), getattr(restored.neural, field)):
                parser.error("Initial checkpoint topology mismatch")
        actor = restored
    interface = make_full_network_actor_interface(model, train_group="neural", input_mode="current")
    stimulation = motor_stimulation_vector(model, args.stimulation_gain)
    commands = command_sequences(args.steps)
    target_fn = ((lambda cmd: traveling_bend_target(cmd, delay=args.propagation_delay, amplitude=args.amplitude))
                 if args.stage == "wave" else (lambda cmd: commanded_bend_target(cmd, amplitude=args.amplitude)))
    metric_fn = traveling_metrics if args.stage == "wave" else alternating_metrics
    pass_fn = traveling_passes if args.stage == "wave" else alternating_passes
    stage_name = "slow_traveling_bend" if args.stage == "wave" else "controlled_alternating_bends"
    targets = jax.vmap(target_fn)(commands)
    def predict(a, cmd):
        return commanded_muscle_sequence(a, interface, cmd, stimulation)
    def objective(a):
        muscles = jax.vmap(lambda cmd: predict(a, cmd))(commands)
        return jnp.mean((muscles - targets) ** 2)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.learning_rate))
    opt_state = optimizer.init(actor)
    @jax.jit
    def update(a, state):
        loss, gradient = jax.value_and_grad(objective)(a)
        updates, state = optimizer.update(gradient, state, a)
        return optax.apply_updates(a, updates), state, loss
    validation_commands = command_sequences(args.steps * 2, held_out=True)
    validation_target = jax.vmap(target_fn)(validation_commands)
    evaluate = jax.jit(lambda a: jax.vmap(metric_fn)(
        jax.vmap(lambda c: predict(a, c))(validation_commands), validation_target))
    def validation(a):
        metrics = evaluate(a)
        if not finite_tree(metrics):
            raise RuntimeError("Nonfinite validation")
        return [{k: float(v[i]) for k, v in metrics.items()} for i in range(len(validation_commands))]
    initial = validation(actor)
    save_actor(output / "actor_initial.npz", actor)
    save_json(output / "manifest.json", dict(stage=stage_name, actor_neurons=302,
        propagation_delay=args.propagation_delay if args.stage == "wave" else None,
        amplitude=args.amplitude,
        input_mode="current", command_interface="artificial_NMJ_projecting_neuron_stimulation",
        timestamp_utc=datetime.now(timezone.utc).isoformat(), train_group="neural", fixed_nmj=True,
        stimulation_gain=args.stimulation_gain, initial_actor=str(args.initial_actor), seed=args.seed,
        initial_actor_sha256=hashlib.sha256(args.initial_actor.read_bytes()).hexdigest() if args.initial_actor else None,
        steps=args.steps, dt=.02, iterations=args.iterations, minutes=args.minutes,
        learning_rate=args.learning_rate, backend=jax.default_backend(),
        jax_version=jax.__version__,
        source_sha256={str(Path(p).resolve().relative_to(repo)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (__file__, motor_curriculum.__file__, full_network_actor.__file__)},
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)),
        body_units="legacy_normalized", muscle_projection="anterior_to_head_at_last_segment", initial=initial))
    best, best_loss, iteration = actor, float("inf"), 0
    status = "completed"
    error = None
    try:
        with (output / "loss.jsonl").open("w", buffering=1) as history:
            while iteration < args.iterations and time.perf_counter() < deadline and not stop:
                candidate, candidate_opt, loss = jax.block_until_ready(update(actor, opt_state))
                if not finite_tree((candidate, candidate_opt, loss)):
                    raise RuntimeError("Nonfinite fitting update; retaining previous finite state")
                if float(loss) < best_loss:
                    best, best_loss = actor, float(loss)
                actor, opt_state = candidate, candidate_opt
                iteration += 1
                history.write(json.dumps(dict(iteration=iteration, mse=float(loss),
                    elapsed_seconds=time.perf_counter() - started)) + "\n")
                if iteration == 1 or iteration % 100 == 0:
                    print(f"iteration={iteration} muscle_mse={float(loss):.7f} elapsed_min={(time.perf_counter()-started)/60:.1f}", flush=True)
                    checkpoint(output / "latest.msgpack", actor, opt_state, jax.random.PRNGKey(args.seed), iteration)
                    save_actor(output / "actor_latest.npz", actor)
        if stop:
            status = "interrupted"
        if iteration == 0:
            raise RuntimeError("Budget expired before training completed an update")
        if float(jax.jit(objective)(actor)) < best_loss:
            best = actor
    except Exception as exception:
        status, error = "failed", repr(exception)
    finally:
        checkpoint(output / "latest.msgpack", actor, opt_state, jax.random.PRNGKey(args.seed), iteration)
        save_actor(output / "actor_final.npz", actor)
        save_actor(output / "actor_best.npz", best)
    final = validation(best)
    passed = all(pass_fn(m) for m in final)
    muscles = jax.jit(jax.vmap(lambda c: predict(best, c)))(validation_commands)
    body = jax.vmap(muscle_sequence_body)(muscles)
    reference = jax.vmap(muscle_sequence_body)(validation_target)
    np.savez_compressed(output / "verification.npz", commands=np.asarray(validation_commands),
        stimulation=np.asarray(stimulation), predicted_muscles=np.asarray(muscles),
        target_muscles=np.asarray(validation_target), joint_angles=np.asarray(body.joint_angles),
        target_joint_angles=np.asarray(reference.joint_angles), position=np.asarray(body.position),
        heading=np.asarray(body.heading), target_position=np.asarray(reference.position),
        target_heading=np.asarray(reference.heading))
    result = dict(stage=stage_name, status=status, error=error, iterations=iteration,
        passed=passed, initial=initial, final=final, elapsed_seconds=time.perf_counter() - started,
        stop_reason="signal" if stop else "time_budget" if time.perf_counter() >= deadline else "iteration_cap",
        next_stage=("locomotion validation" if args.stage == "wave" else "slow traveling wave") if passed else "stay on current motor stage",
        limitations="Artificial neural stimulation; no autonomous oscillator or locomotion demonstrated")
    save_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed and status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
