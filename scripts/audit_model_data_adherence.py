"""Model-versus-data audit: does the network start from what the data say, and where is it only a placeholder?

Sections: S1 topology against the raw Cook 2019 sheets; S2 synapse reversal against Wang 2024 transmitter labels;
S3 provenance of single-cell conductances; S4 parameters sitting on migration clip bounds; S5 resting-state
transmission; S6 resting voltage against the digitized reference traces; S7 structural/peptide control predictors
against the atlas statistics (the same metrics used for the model in E100). Read-only; nothing is trained.
"""
import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from r_elegans.assets import load_asset_document
from r_elegans.atlas_evaluation import model_pairs, pair_metrics
from r_elegans.brain.conductance_network import load_network, network_synapses
from r_elegans.brain.single_compartment import CHANNEL_INDEX
from r_elegans.brain.stimulation_response import settle
from r_elegans.data.connectome import load_connectome
from r_elegans.data.neuromuscular import parse_wang_neurotransmitter_workbook
from r_elegans.data.physiology import load_electrophysiology_trace, load_neuron_parameters

warnings.filterwarnings('ignore')
NEURONS = list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
INDEX = {n: i for i, n in enumerate(NEURONS)}


def read_cook(path, sheet):
    from openpyxl import load_workbook
    rows = list(load_workbook(path, read_only=True, data_only=True)[sheet].iter_rows(values_only=True))
    columns = {name: j for j, name in enumerate(rows[2]) if isinstance(name, str) and j >= 3}
    full = {}
    for row in rows[3:]:
        pre = row[2]
        if not isinstance(pre, str):
            continue
        for name, j in columns.items():
            if j < len(row) and isinstance(row[j], (int, float)) and row[j]:
                full[(pre, name)] = float(row[j])
    return full


def matrix_from(full, symmetric=False):
    """[post, pre] over the 302 model neurons, plus names absent from the sheet."""
    matrix = np.zeros((302, 302))
    for (pre, post), value in full.items():
        if pre in INDEX and post in INDEX:
            matrix[INDEX[post], INDEX[pre]] = value
    sheet_names = {n for key in full for n in key}
    return matrix, sorted(n for n in NEURONS if n not in sheet_names)


def s1_topology(root):
    path = root / 'raw/connectome/cook2019/41586_2019_1352_MOESM9_ESM.xlsx'
    connectome = load_connectome()
    chem_model, gap_model = np.asarray(connectome.chemical_counts), np.asarray(connectome.gap_counts)
    out = {}
    for label, sheet, model in (('chemical', 'hermaphrodite chemical', chem_model), ('gap', 'herm gap jn symmetric', gap_model)):
        raw, missing = matrix_from(read_cook(path, sheet))
        different = np.argwhere(raw != model)
        out[label] = dict(model_edges=int((model > 0).sum()), raw_edges=int((raw > 0).sum()), model_total=float(model.sum()), raw_total=float(raw.sum()),
                          entries_that_differ=int(len(different)), names_missing_in_sheet=missing,
                          examples=[(NEURONS[p], NEURONS[q], float(raw[p, q]), float(model[p, q])) for p, q in different[:8]])
    asym, _ = matrix_from(read_cook(path, 'herm gap jn asymmetric'))
    out['gap_raw_asymmetric_pairs_unequal'] = int(np.sum(np.triu(asym != asym.T, 1)))
    return out


def reversal_category(value):
    return 'excitatory' if value > -40 else 'inhibitory' if value < -60 else 'intermediate'


def s2_polarity(root, fixed, chemical):
    labels = parse_wang_neurotransmitter_workbook(root=root)
    table, weighted = Counter(), Counter()
    outgoing = chemical.sum(axis=0)
    for i, name in enumerate(NEURONS):
        base = re.match(r'[A-Z]+', name)
        label = labels.get(name, labels.get(f'class:{base.group(0)}' if base else '', 'unlabeled'))
        category = reversal_category(float(fixed.synapse_reversal_mV[i]))
        table[(label, category)] += 1
        weighted[(label, category)] += float(outgoing[i])
    return dict(counts={f'{k[0]} | {k[1]}': v for k, v in sorted(table.items())},
                outgoing_weight={f'{k[0]} | {k[1]}': round(v, 4) for k, v in sorted(weighted.items())},
                gaba_only_not_inhibitory=sum(v for (l, c), v in table.items() if l.lower() == 'gaba' and c != 'inhibitory'),
                ach_only_not_excitatory=sum(v for (l, c), v in table.items() if l.lower() == 'ach' and c != 'excitatory'),
                reversal_values=Counter(np.round(np.asarray(fixed.synapse_reversal_mV), 1).tolist()).most_common(6))


def s3_cells(params, fixed, config):
    awc, rmd = (np.asarray(load_neuron_parameters(c).params.conductances_nS) for c in ('AWCON', 'RMD'))
    g, kinds = np.asarray(fixed.cells.conductances_nS), Counter()
    leak = CHANNEL_INDEX['LEAK']  # overwritten to the synaptic unit by construction
    mask = np.ones(g.shape[1], bool)
    mask[leak] = False
    for i, name in enumerate(NEURONS):
        if name.startswith('RMD'):
            kinds['RMD (measured values, rhythm prior may modify)'] += 1
        elif np.allclose(np.maximum(awc[mask] * config.transferred_channel_scale, 1e-5), g[i, mask], rtol=1e-6):
            kinds['AWC composition x transferred scale (placeholder)'] += 1
        else:
            kinds['other'] += 1
    scale = np.asarray(params.log_channel_scale)
    evidence = {c: dict(Counter(load_neuron_parameters(c).evidence.values())) for c in ('AWCON', 'RMD')}
    return dict(cell_kinds=dict(kinds), transferred_channel_scale=config.transferred_channel_scale,
                trained_channel_scale_changed_cells=int((np.abs(scale).max(axis=1) > 1e-6).sum()),
                trained_channel_scale_abs_log_median=float(np.median(np.abs(scale))), trained_channel_scale_abs_log_max=float(np.abs(scale).max()),
                evidence_labels_in_measured_classes=evidence)


def s4_clip_bounds(fixed):
    def at(values, low, high, tol=1e-6):
        v = np.asarray(values)
        return dict(at_low=int((np.abs(v - low) < tol).sum()), at_high=int((np.abs(v - high) < tol).sum()))
    return dict(initial_voltage_mV=at(fixed.initial_voltage_mV, -80., -35.), leak_reversal_mV=at(fixed.cells.leak_reversal_mV, -80., -35.),
                release_threshold_mV=at(fixed.release_threshold_mV, -65., -20.), release_slope_mV=at(fixed.release_slope_mV, 2., 20.),
                synapse_reversal_mV=at(fixed.synapse_reversal_mV, -85., 0.), capacitance_pF=at(fixed.cells.capacitance_pF, 1., 20.),
                neurons=302)


def s5_resting_transmission(params, fixed, config):
    state = settle(params, fixed, config, warmup_s=60., dt_s=.1, substeps=20)
    v = np.asarray(state.voltage_mV)
    theta, slope = np.asarray(fixed.release_threshold_mV), np.asarray(fixed.release_slope_mV)
    release = 1. / (1. + np.exp(-(v - theta) / slope))
    gain = release * (1. - release) / slope  # d release / dV, per mV
    chemical, _ = network_synapses(params, fixed, config)
    chemical = np.asarray(chemical)
    edge_gain = chemical * gain[None, :]
    leak = np.asarray(fixed.cells.conductances_nS)[:, CHANNEL_INDEX['LEAK']]
    return dict(voltage_mV=dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max())),
                release_below_0p01=int((release < .01).sum()), release_above_0p99=int((release > .99).sum()), release_between=int(((release >= .01) & (release <= .99)).sum()),
                median_release_gain_per_mV=float(np.median(gain)), neurons_with_gain_above_0p01_per_mV=int((gain > .01).sum()),
                chemical_weight_total_nS=float(chemical.sum()), chemical_weight_with_presynaptic_gain_below_1e3=float(chemical[:, gain < 1e-3].sum() / max(chemical.sum(), 1e-30)),
                median_edge_sensitivity_over_postsynaptic_leak=float(np.median((edge_gain / leak[:, None])[chemical > 0])),
                fraction_of_edges_with_sensitivity_over_leak_above_1e3=float(np.mean((edge_gain / leak[:, None])[chemical > 0] > 1e-3)),
                voltage_by_neuron={n: round(float(v[INDEX[n]]), 2) for n in ('AWAL', 'AWAR', 'AIYL', 'AIYR', 'RIML', 'AVAL', 'AVBL', 'RIAL', 'RIAR', 'RMDDL', 'SMDDL')}), v


def s6_rest_traces(root, settled_v):
    out = {}
    for name in ('AVAL', 'AIYL', 'RIML', 'AWAL'):
        trace = load_electrophysiology_trace(f'raw/physiology/baaiworm2024/{name}_Current_Clamp_Trace.txt', root=root)
        early = trace.responses[(trace.time_s <= trace.time_s[0] + .1)].mean(axis=0) * 1e3
        out[name] = dict(trace_first_100ms_mV=[round(float(x), 1) for x in early[:4]], trace_unit=trace.response_unit,
                         model_rest_mV=round(float(settled_v[INDEX[name]]), 1))
    return out


def s7_controls(root, seed):
    from scripts.run_atlas_baseline import build_statistics
    _, _, train, test, *_ = build_statistics(root / 'raw/functional/randi2023_dandi001075_segmentation', NEURONS, seed)
    connectome = load_connectome()
    chem, gap = np.asarray(connectome.chemical_counts), np.asarray(connectome.gap_counts)
    adjacency = ((chem + gap) > 0).astype(float)
    predictors = dict(direct_chemical=chem, direct_gap=gap, two_step_walks=adjacency @ adjacency, three_step_walks=adjacency @ adjacency @ adjacency,
                      responder_in_degree=np.tile((chem > 0).sum(axis=1)[:, None], (1, 302)),
                      stimulated_out_degree=np.tile((chem > 0).sum(axis=0)[None, :], (302, 1)))
    candidate = root / 'processed/extrasynaptic_candidates_v1/candidate_pairs_cengen2_ec50le500nM.csv.gz'
    if candidate.exists():
        import pandas as pd
        frame = pd.read_csv(candidate)
        peptide = np.zeros((302, 302))
        peptide[[INDEX[r] for r in frame.receiver], [INDEX[s] for s in frame.sender]] = frame.n_pairs
        predictors['peptide_candidate_pairs'] = peptide
    result = {}
    shared = sorted(set(train['pairs']) & set(test['pairs']))
    ti, ei = {p: i for i, p in enumerate(train['pairs'])}, {p: i for i, p in enumerate(test['pairs'])}
    a = np.array([train['mean_z'][ti[p]] for p in shared])
    b = np.array([test['mean_z'][ei[p]] for p in shared])
    from scipy.stats import spearmanr
    result['data_ceiling_train_vs_test_pair_mean_z'] = dict(shared_pairs=len(shared), spearman=round(float(spearmanr(a, b)[0]), 4))
    for name, matrix in predictors.items():
        result[name] = {}
        for label, stats in (('train', train), ('test', test)):
            values = np.array([matrix[INDEX[r], INDEX[t]] for t, r in stats['pairs']], float)
            scores = pair_metrics(values, stats, permutations=300, seed=seed)
            result[name][label] = {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=Path, default=Path(__file__).resolve().parents[1] / 'r_elegans/assets/mac_rhythm_v1/network.npz')
    parser.add_argument('--data-root', type=Path, default=Path.home() / 'OneDrive/r-elegans')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--skip-controls', action='store_true')
    parser.add_argument('--run', type=Path, default=None, help='with --checkpoint-sha256: find network.npz by hash under this run')
    parser.add_argument('--checkpoint-sha256', default=None)
    args = parser.parse_args()
    if args.run and args.checkpoint_sha256:
        from scripts.run_atlas_baseline import find_checkpoint
        args.network = find_checkpoint(args.run, args.checkpoint_sha256)
    params, fixed, config = load_network(args.network)
    to64 = lambda a: jnp.asarray(a, jnp.float64) if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a
    params, fixed = jax.tree_util.tree_map(to64, (params, fixed))  # older checkpoints were saved as float32
    chemical = np.asarray(network_synapses(params, fixed, config)[0])
    report = dict(network=str(args.network), s1_topology_vs_cook=s1_topology(args.data_root), s2_polarity_vs_wang=s2_polarity(args.data_root, fixed, chemical),
                  s3_cell_provenance=s3_cells(params, fixed, config), s4_clip_bound_pileup=s4_clip_bounds(fixed))
    resting, settled_v = s5_resting_transmission(params, fixed, config)
    report['s5_resting_transmission'] = resting
    report['s6_rest_voltage_vs_traces'] = s6_rest_traces(args.data_root, settled_v)
    if not args.skip_controls:
        report['s7_atlas_controls'] = s7_controls(args.data_root, 0)
    text = json.dumps(report, indent=1, default=float)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == '__main__':
    main()
