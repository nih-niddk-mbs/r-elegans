"""Compare our trained connectome subcircuit's synaptic signs against real data.

Analysis only -- this script does not train or fine-tune anything. It loads
an already-trained checkpoint (default: the RL-fine-tuned connectome
checkpoint saved by ``scripts/train_rl_chemotaxis.py --actor connectome``)
and compares its learned per-neuron ``synapse_reversal`` (this repo's
parameterization of a neuron's excitatory/inhibitory character -- see
``r_elegans.brain.circuit.RawSubcircuitParams``) against real, per-edge sign
predictions from:

    Fenyves, B. G., Szilagyi, G. S., Vassy, Z., Soti, C., & Csermely, P.
    (2020). Synaptic polarity and sign-balance prediction using gene
    expression data in the Caenorhabditis elegans chemical synapse neuronal
    connectome network. PLOS Computational Biology, 16(12), e1007974.
    https://doi.org/10.1371/journal.pcbi.1007974 (CC BY 4.0)

See ``scripts/data/fenyves2020_subcircuit_signs.csv`` for the full
extraction provenance. Cook et al. 2019 (the connectome topology already
bundled in this repo) contains no synapse-sign data of its own -- this is
the reason a second, independent source is needed at all.

Usage:
    python scripts/compare_synapse_signs.py
    python scripts/compare_synapse_signs.py --checkpoint PATH --output PATH
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from r_elegans.brain import SUBCIRCUIT_NEURON_NAMES, decode_subcircuit_params
from r_elegans.rl import actor_from_arrays

DEFAULT_CHECKPOINT = Path("results/behavior/petri_chemotaxis_connectome_ppo_v1.npz")
DEFAULT_SIGN_DATA = Path("scripts/data/fenyves2020_subcircuit_signs.csv")
DEFAULT_OUTPUT = Path("results/analysis/synapse_sign_comparison.png")

CIRCUIT_GROUPS: dict[str, str] = {
    "AWC": "sensory", "ASE": "sensory",
    "AIY": "interneuron 1", "AIZ": "interneuron 2",
    "RIA": "integrator", "RMDD": "motor (dorsal)", "RMDV": "motor (ventral)",
}


def load_sign_predictions(path: Path) -> list[dict]:
    """Parse the ``#``-commented CSV of real per-edge sign predictions."""

    with path.open(newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines))


def per_neuron_real_polarity(rows: list[dict]) -> dict[str, dict]:
    """Aggregate each presynaptic neuron's net signed weight and confidence.

    ``polarity_fraction`` is ``net_signed_weight / resolved_weight`` in
    ``[-1, 1]``: +1 means every cleanly-predicted outgoing edge is
    excitatory, -1 means every one is inhibitory, and intermediate values
    (e.g. AIYR) mean the predictions genuinely disagree across edges, not
    that data is missing. ``resolved_fraction`` separately reports how much
    of the neuron's total contact weight got a clean +/- prediction at all
    (vs. "complex" or "no pred").
    """

    totals = {name: {"net": 0, "total": 0, "resolved": 0} for name in SUBCIRCUIT_NEURON_NAMES}
    for row in rows:
        pre = row["presynaptic"]
        weight = int(row["contact_weight"])
        sign = row["predicted_sign"]
        entry = totals[pre]
        entry["total"] += weight
        if sign == "+":
            entry["net"] += weight
            entry["resolved"] += weight
        elif sign == "-":
            entry["net"] -= weight
            entry["resolved"] += weight

    summary = {}
    for name, entry in totals.items():
        resolved = entry["resolved"]
        summary[name] = {
            "polarity_fraction": (entry["net"] / resolved) if resolved > 0 else None,
            "resolved_fraction": (resolved / entry["total"]) if entry["total"] > 0 else 0.0,
            "total_weight": entry["total"],
        }
    return summary


def build_records(checkpoint: Path, sign_data: Path) -> list[dict]:
    with np.load(checkpoint, allow_pickle=False) as data:
        actor = actor_from_arrays(data)
    neural_params = decode_subcircuit_params(actor.neural_params)
    trained_reversal = np.asarray(neural_params.synapse_reversal)
    trained_rest = np.asarray(neural_params.leak_reversal)

    real = per_neuron_real_polarity(load_sign_predictions(sign_data))

    records = []
    for index, name in enumerate(SUBCIRCUIT_NEURON_NAMES):
        delta = float(trained_reversal[index] - trained_rest[index])
        entry = real[name]
        records.append(
            {
                "name": name,
                "group": CIRCUIT_GROUPS[name[:-1]],
                "trained_delta": delta,
                "trained_sign": "excitatory" if delta > 0 else "inhibitory",
                "real_polarity_fraction": entry["polarity_fraction"],
                "real_resolved_fraction": entry["resolved_fraction"],
                "real_total_weight": entry["total_weight"],
            }
        )
    return records


def print_summary(records: list[dict]) -> None:
    header = (
        f"{'neuron':7s} {'group':16s} {'trained Δreversal':>17s} {'trained sign':>13s}  "
        f"{'real polarity':>13s} {'real resolved%':>15s}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        real_polarity = (
            f"{r['real_polarity_fraction']:+.2f}" if r["real_polarity_fraction"] is not None else "no data"
        )
        print(
            f"{r['name']:7s} {r['group']:16s} {r['trained_delta']:17.3f} {r['trained_sign']:>13s}  "
            f"{real_polarity:>13s} {100 * r['real_resolved_fraction']:14.0f}%"
        )

    resolvable = [r for r in records if r["real_polarity_fraction"] is not None]
    agree = [
        r for r in resolvable
        if (r["real_polarity_fraction"] > 0) == (r["trained_delta"] > 0)
    ]
    print(
        f"\n{len(agree)}/{len(resolvable)} neurons with a real-data prediction agree in "
        "sign with the trained model "
        f"({len(records) - len(resolvable)} neuron(s) have no resolved real-data prediction)."
    )


def make_figure(records: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    excitatory_color = "#277da1"
    inhibitory_color = "#c44536"
    unresolved_color = "#9a9a93"

    names = [r["name"] for r in records]
    y = np.arange(len(names))[::-1]  # top-to-bottom in subcircuit order

    fig, (ax_real, ax_trained) = plt.subplots(1, 2, figsize=(11.0, 6.0), sharey=True)

    # --- Left: real data (Fenyves et al. 2020) ---
    for yi, r in zip(y, records):
        fraction = r["real_polarity_fraction"]
        if fraction is None:
            ax_real.plot(0, yi, marker="x", color=unresolved_color, markersize=8, markeredgewidth=2)
            ax_real.text(0.05, yi, "no resolved prediction", va="center", fontsize=8, color=unresolved_color)
            continue
        color = excitatory_color if fraction > 0 else inhibitory_color
        alpha = 0.35 + 0.65 * r["real_resolved_fraction"]
        ax_real.barh(yi, fraction, height=0.6, color=color, alpha=alpha)
    ax_real.axvline(0.0, color="#55554f", linewidth=1.0)
    ax_real.set_xlim(-1.15, 1.15)
    ax_real.set_xlabel("net predicted polarity\n(fraction of resolved contact weight)")
    ax_real.set_title("Real data\n(Fenyves et al. 2020)")

    # --- Right: our trained model ---
    for yi, r in zip(y, records):
        delta = r["trained_delta"]
        color = excitatory_color if delta > 0 else inhibitory_color
        ax_trained.barh(yi, delta, height=0.6, color=color)
    ax_trained.axvline(0.0, color="#55554f", linewidth=1.0)
    ax_trained.set_xlabel("trained reversal potential − resting potential\n(model units)")
    ax_trained.set_title("Our trained model\n(synapse_reversal)")

    ax_real.set_yticks(y)
    ax_real.set_yticklabels(names)
    ax_real.set_ylim(-0.8, len(names) - 0.2)

    for ax in (ax_real, ax_trained):
        ax.set_facecolor("#fffdf6")
        ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=excitatory_color, label="Excitatory"),
        plt.Rectangle((0, 0), 1, 1, color=inhibitory_color, label="Inhibitory"),
        plt.Line2D([0], [0], marker="x", color=unresolved_color, linestyle="none",
                   markeredgewidth=2, markersize=8, label="No resolved real-data prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        "Predicted vs. trained synaptic sign: 14-neuron chemotaxis subcircuit",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"\nfigure written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sign-data", type=Path, default=DEFAULT_SIGN_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "This script only analyzes an already-trained model; it does not "
            "train one. Run scripts/train_rl_chemotaxis.py --actor connectome "
            "first, or pass --checkpoint."
        )

    records = build_records(args.checkpoint, args.sign_data)
    print_summary(records)
    make_figure(records, args.output)


if __name__ == "__main__":
    main()
