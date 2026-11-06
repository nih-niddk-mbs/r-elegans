"""Frozen full-brain go/stop/restart assay; persistent brain and body state."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.body import initialize_muscle_body, default_muscle_body_params
from r_elegans.model import load_builtin_model
from r_elegans.rl.full_network_actor import actor_from_arrays, init_full_network_actor, make_full_network_actor_interface
from r_elegans.rl.motor_curriculum import (directional_muscle_sequence, reverse_context_vector,
    motor_stimulation_vector, muscle_sequence_body)
from r_elegans.rl import stop_restart
from r_elegans.rl.stop_restart import stop_restart_commands, stop_restart_metrics, assess_stop_restart
if __package__:
    from .experiment_records import capture_run_record
    from .train_timed import save_json, finite_tree
else:
    from experiment_records import capture_run_record
    from train_timed import save_json, finite_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == repo or repo in output.parents:
        parser.error("Artifacts must stay outside Git")
    source = json.loads((args.initial_actor.parent / "manifest.json").read_text())
    if (source.get("input_mode") != "current" or source.get("body_units") != "legacy_normalized"
            or source.get("muscle_projection") != "anterior_to_head_at_last_segment"):
        parser.error("Requires a normalized, head-aligned current-input checkpoint")
    gain = source["stimulation_gain"]
    context_gain = source.get("reverse_context_gain")
    if args.direction == "reverse" and (source.get("reverse_context_scheme") != "fixed_random_NMJ_neurons_seed719_v1"
            or context_gain is None or not math.isfinite(context_gain) or context_gain <= 0):
        parser.error("Reverse assay requires a checkpoint fitted with the matching direction context")
    if not math.isfinite(gain) or gain <= 0:
        parser.error("Stimulation gain must be finite and positive")
    model = load_builtin_model()
    with np.load(args.initial_actor, allow_pickle=False) as data:
        actor = actor_from_arrays(data)
    expected = init_full_network_actor(model, jax.random.PRNGKey(0)).neural
    for name in ("chemical_mask", "gap_mask"):
        if not np.array_equal(getattr(actor.neural, name), getattr(expected, name)):
            parser.error("Checkpoint topology mismatch")
    output.mkdir(parents=True, exist_ok=False)
    capture_run_record(output, repo, args)
    started = time.perf_counter()
    save_json(output / "manifest.json", dict(stage="stop_restart_assay", training=False,
        actor_neurons=302, seed=args.seed, initial_actor=str(args.initial_actor),
        direction=args.direction, reverse_context_gain=context_gain,
        initial_actor_sha256=hashlib.sha256(args.initial_actor.read_bytes()).hexdigest(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(), backend=jax.default_backend(),
        stimulation_gain=gain, command_amplitude=.85, periods_seconds=[2.4, 3., 3.7],
        warmup_cycles_per_block=3, measured_cycles_per_block=3, state_resets_between_blocks=False,
        dt_seconds=[.02, .01], controls=["continuous_zero_command_stop", "isotropic_drag"],
        maximum_stop_motion_ratio=.1, restart_progress_ratio_range=[.8, 1.2],
        minimum_progress_body_lengths_per_cycle=.001, body_units="legacy_normalized",
        source_sha256={str(Path(p).resolve().relative_to(repo)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (__file__, stop_restart.__file__)},
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True))))
    stimulation = motor_stimulation_vector(model, gain)
    context = reverse_context_vector(model, context_gain or .05)
    results = []
    for period in (2.4, 3., 3.7):
        reports = {}
        for dt, isotropic in ((.02, False), (.01, False), (.01, True)):
            period_steps = round(period / dt)
            commands = jnp.asarray(stop_restart_commands(period_steps))
            interface = make_full_network_actor_interface(model, dt=dt, input_mode="current")
            params = default_muscle_body_params(12)
            if isotropic:
                params = params._replace(mechanics=params.mechanics._replace(
                    perpendicular_drag=params.mechanics.parallel_drag))
            @jax.jit
            def simulate():
                muscles = directional_muscle_sequence(actor, interface, commands, stimulation,
                    context=context, reverse=args.direction == "reverse")
                return muscle_sequence_body(muscles, dt=dt, body_params=params)
            body = jax.block_until_ready(simulate())
            if not finite_tree(body):
                raise RuntimeError("Nonfinite stop/restart trajectory")
            initial = initialize_muscle_body(12)
            states = jax.tree_util.tree_map(lambda a, b: np.concatenate([np.asarray(a)[None], np.asarray(b)]), initial, body)
            label = "isotropic" if isotropic else "fine" if dt == .01 else "coarse"
            reports[label] = stop_restart_metrics(states.position, states.heading,
                states.joint_angles, period_steps=period_steps, dt=dt)
            np.savez_compressed(output / f"period-{period}-{label}.npz", commands=np.asarray(commands),
                time=states.time, position=states.position, heading=states.heading,
                joint_angles=states.joint_angles, muscle_activation=states.muscle_activation)
        result = dict(period_seconds=period, **assess_stop_restart(
            reports["coarse"], reports["fine"], reports["isotropic"], direction=args.direction), cycles=reports)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "cycles"}), flush=True)
    passed = all(r["passed"] for r in results)
    summary = dict(stage="stop_restart_assay", direction=args.direction, status="completed", error=None, passed=passed,
        control_passed=all(r["control_passed"] for r in results), periods=results,
        elapsed_seconds=time.perf_counter()-started,
        limitations="Artificial pacing, normalized body; late settling windows, not biological stopping latency",
        next_stage="reverse-motion primitive" if passed else "inspect stop/restart and propulsion gates before training")
    save_json(output / "summary.json", summary)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
