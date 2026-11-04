"""Train and verify only a small static bend in the full recurrent network.

No RL, food, gait phase, teacher voltage, or periodic input. NMJ/muscle transfer
parameters stay fixed so the bend must be learned in the recurrent brain.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import jax
import numpy as np
import optax

from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import init_full_network_actor, make_full_network_actor_interface
from r_elegans.rl.motor_curriculum import (
    bend_target, neural_muscle_sequence, muscle_sequence_body, make_bend_update,
    bend_metrics, bend_passes,
)
from r_elegans.rl import motor_curriculum
if __package__:
    from .train_timed import finite_tree, save_actor, save_json, checkpoint
else:
    from train_timed import finite_tree, save_actor, save_json, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--amplitude", type=float, default=0.04)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.iterations < 1 or args.steps < 2 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Positive iterations, learning rate and at least two steps required")
    if not 0 < args.amplitude <= 0.1:
        parser.error("Amplitude must be in (0, 0.1]")
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == repo or repo in output.parents:
        parser.error("Keep artifacts outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = load_builtin_model()
    actor = init_full_network_actor(model, jax.random.PRNGKey(args.seed))
    interface = make_full_network_actor_interface(model, train_group="neural")
    target = bend_target(args.steps, amplitude=args.amplitude)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate))
    opt_state = optimizer.init(actor)
    update = make_bend_update(interface, target, optimizer)
    evaluate = jax.jit(lambda a: bend_metrics(neural_muscle_sequence(a, interface, args.steps * 2),
                                             bend_target(args.steps * 2, amplitude=args.amplitude)))
    def metrics(a):
        values = evaluate(a)
        if not finite_tree(values):
            raise RuntimeError("Nonfinite evaluation")
        return {name: float(value) for name, value in values.items()}
    initial = metrics(actor)
    save_actor(output / "actor_initial.npz", actor)
    save_json(output / "manifest.json", dict(stage="small_static_bend", actor_neurons=302,
        train_group="neural", fixed_nmj=True, input="zero", seed=args.seed, dt=0.02,
        training_steps=args.steps, evaluation_steps=args.steps * 2, amplitude=args.amplitude,
        iterations=args.iterations, learning_rate=args.learning_rate, body_units="legacy_normalized",
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)),
        source_sha256={str(Path(p).resolve().relative_to(repo)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (__file__, motor_curriculum.__file__)},
        jax_version=jax.__version__,
        backend=jax.default_backend(), initial=initial))
    best = actor
    best_loss = float("inf")
    with (output / "loss.jsonl").open("w", buffering=1) as history:
        for iteration in range(1, args.iterations + 1):
            candidate, candidate_opt, loss = jax.block_until_ready(update(actor, opt_state))
            if not finite_tree((candidate, candidate_opt, loss)):
                raise RuntimeError("Nonfinite fitting update; previous checkpoint retained")
            # Loss belongs to the pre-update actor.
            if float(loss) < best_loss:
                best, best_loss = actor, float(loss)
            actor, opt_state = candidate, candidate_opt
            history.write(json.dumps(dict(iteration=iteration, mse=float(loss))) + "\n")
            if iteration == 1 or iteration % 50 == 0 or iteration == args.iterations:
                print(f"iteration={iteration} muscle_mse={float(loss):.7f}", flush=True)
                checkpoint(output / "latest.msgpack", actor, opt_state, jax.random.PRNGKey(args.seed), iteration)
    # Include the last update when selecting by training loss.
    final_sequence = neural_muscle_sequence(actor, interface, args.steps)
    final_loss = float(np.mean((np.asarray(final_sequence) - np.asarray(target)) ** 2))
    if final_loss < best_loss:
        best = actor
    final = metrics(best)
    passed = bool(bend_passes(final))
    save_actor(output / "actor_final.npz", actor)
    save_actor(output / "actor_best.npz", best)
    muscles = neural_muscle_sequence(best, interface, args.steps * 2)
    body = muscle_sequence_body(muscles)
    teacher = bend_target(args.steps * 2, amplitude=args.amplitude)
    teacher_body = muscle_sequence_body(teacher)
    np.savez_compressed(output / "verification.npz", predicted_muscles=np.asarray(muscles),
        target_muscles=np.asarray(teacher), joint_angles=np.asarray(body.joint_angles),
        target_joint_angles=np.asarray(teacher_body.joint_angles), position=np.asarray(body.position))
    result = dict(stage="small_static_bend", initial=initial, final=final, passed=passed,
        elapsed_seconds=time.perf_counter() - started,
        next_stage="review bend before attempting alternating bends" if passed else "stay on bend fitting",
        limitations="One autonomous bend in normalized mechanics; no command control or locomotion demonstrated")
    save_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
