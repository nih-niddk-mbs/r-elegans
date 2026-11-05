"""Measure steady locomotion of a fitted 302-neuron wave policy. No training."""
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
from r_elegans.rl.motor_curriculum import commanded_muscle_sequence, motor_stimulation_vector, muscle_sequence_body
from r_elegans.rl.locomotion import cycle_metrics, assess_propulsion
if __package__:
    from .train_timed import save_json, finite_tree
else:
    from train_timed import save_json, finite_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-actor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-cycles", type=int, default=4)
    parser.add_argument("--measured-cycles", type=int, default=6)
    args = parser.parse_args()
    if args.warmup_cycles < 2 or args.measured_cycles < 3:
        parser.error("At least two warmup cycles and three measured cycles required")
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    if output == repo or repo in output.parents:
        parser.error("Artifacts must stay outside Git")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = load_builtin_model()
    source_manifest = json.loads((args.initial_actor.parent / "manifest.json").read_text())
    if (source_manifest.get("input_mode") != "current"
            or source_manifest.get("body_units") != "legacy_normalized"
            or source_manifest.get("muscle_projection") != "anterior_to_head_at_last_segment"):
        parser.error("Assay requires the matching normalized, head-aligned current-input checkpoint manifest")
    gain = source_manifest["stimulation_gain"]
    if not math.isfinite(gain) or gain <= 0:
        parser.error("Checkpoint stimulation gain must be finite and positive")
    with np.load(args.initial_actor, allow_pickle=False) as data:
        actor = actor_from_arrays(data)
    expected = init_full_network_actor(model, jax.random.PRNGKey(0)).neural
    for name in ("chemical_mask", "gap_mask"):
        if not np.array_equal(getattr(actor.neural, name), getattr(expected, name)):
            parser.error("Checkpoint topology mismatch")
    stimulation = motor_stimulation_vector(model, gain)
    manifest = dict(stage="locomotion_assay", seed=args.seed, actor_neurons=302, training=False,
        timestamp_utc=datetime.now(timezone.utc).isoformat(), backend=jax.default_backend(),
        initial_actor=str(args.initial_actor), initial_actor_sha256=hashlib.sha256(args.initial_actor.read_bytes()).hexdigest(),
        warmup_cycles=args.warmup_cycles, measured_cycles=args.measured_cycles,
        periods_seconds=[2.4, 3.0, 3.7], dt_seconds=[.02, .01], command_amplitude=.85,
        controls=["zero_neural_command", "isotropic_drag"], minimum_useful_progress_body_lengths_per_cycle=.001,
        body_units="legacy_normalized", input_mode="artificial_neural_current", body_length=1.,
        stimulation_gain=gain,
        projection="anterior_to_head_at_last_segment",
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)))
    save_json(output / "manifest.json", manifest)
    results = []
    for period in manifest["periods_seconds"]:
        reports = {}
        for dt, rest, isotropic in ((.02, False, False), (.01, False, False), (.01, True, False), (.01, False, True)):
            period_steps = round(period / dt)
            steps = period_steps * (args.warmup_cycles + args.measured_cycles)
            interface = make_full_network_actor_interface(model, dt=dt, input_mode="current")
            commands = jnp.zeros(steps) if rest else .85 * jnp.sin(2*jnp.pi*jnp.arange(steps)/period_steps)
            body_params = default_muscle_body_params(12)
            if isotropic:
                body_params = body_params._replace(mechanics=body_params.mechanics._replace(
                    perpendicular_drag=body_params.mechanics.parallel_drag))
            @jax.jit
            def simulate():
                muscles = commanded_muscle_sequence(actor, interface, commands, stimulation)
                return muscle_sequence_body(muscles, dt=dt, body_params=body_params)
            body = jax.block_until_ready(simulate())
            if not finite_tree(body):
                raise RuntimeError("Nonfinite locomotion trajectory")
            initial = initialize_muscle_body(12)
            states = jax.tree_util.tree_map(lambda a, b: np.concatenate([np.asarray(a)[None], np.asarray(b)]), initial, body)
            metrics = cycle_metrics(states.position, states.heading, states.joint_angles,
                period_steps=period_steps, warmup_cycles=args.warmup_cycles, measured_cycles=args.measured_cycles)
            label = "isotropic" if isotropic else "rest" if rest else "fine" if dt == .01 else "coarse"
            reports[label] = metrics
            np.savez_compressed(output / f"period-{period}-{label}.npz",
                time=np.asarray(states.time), position=states.position, heading=states.heading,
                joint_angles=states.joint_angles, muscle_activation=states.muscle_activation)
        decision = assess_propulsion(reports["coarse"], reports["fine"], reports["rest"], isotropic=reports["isotropic"])
        results.append(dict(period_seconds=period, **decision, cycles=reports))
        print(json.dumps({k: v for k, v in results[-1].items() if k != "cycles"}), flush=True)
    passed = all(r["passed"] for r in results)
    summary = dict(stage="locomotion_assay", status="completed", error=None, passed=passed,
        elapsed_seconds=time.perf_counter()-started, periods=results,
        next_stage="direction and turning primitives" if passed else "review propulsion, then increase motor targets gradually",
        limitations="Externally paced, normalized body; propulsion thresholds are engineering criteria, not biological fidelity")
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
