"""Candidate extrasynaptic peptide edges: CeNGEN expression x Beets et al. 2023 peptide-GPCR EC50s.

A sender neuron is a candidate for a receiver when the sender expresses a peptide precursor gene, the receiver
expresses a GPCR gene, and the measured EC50 for that peptide-receptor pair is at or below the cutoff.
Signs, strengths and dynamics are not inferred here.
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from r_elegans.assets import load_asset_document

warnings.filterwarnings('ignore')
EXPLICIT = {'DA09': ['DA9'], 'VC04': ['VC_4_5'], 'VC05': ['VC_4_5'], 'AWCL': ['AWC_ON', 'AWC_OFF'], 'AWCR': ['AWC_ON', 'AWC_OFF'],
            'IL2DL': ['IL2_DV'], 'IL2DR': ['IL2_DV'], 'IL2VL': ['IL2_DV'], 'IL2VR': ['IL2_DV'], 'IL2L': ['IL2_LR'], 'IL2R': ['IL2_LR'],
            'RMDDL': ['RMD_DV'], 'RMDDR': ['RMD_DV'], 'RMDVL': ['RMD_DV'], 'RMDVR': ['RMD_DV'], 'RMDL': ['RMD_LR'], 'RMDR': ['RMD_LR'],
            'RMED': ['RME_DV'], 'RMEV': ['RME_DV'], 'RMEL': ['RME_LR'], 'RMER': ['RME_LR']}
EXPLICIT.update({f'DD{i:02d}': ['VD_DD'] for i in range(1, 7)})
EXPLICIT.update({f'VD{i:02d}': ['VD_DD'] for i in range(1, 14)})
SUFFIX = r'(L|R|DL|DR|VL|VR|D|V)?\d*'
AMBIGUOUS = ('AWCL', 'AWCR')


def class_assignment(neuron_ids, classes):
    """Map each neuron to its most specific CeNGEN class(es); report unmapped neurons and unused classes."""
    assignment = {}
    for neuron in neuron_ids:
        if neuron in EXPLICIT:
            assignment[neuron] = [c for c in EXPLICIT[neuron] if c in classes]
            continue
        options = [c for c in classes if re.fullmatch(re.escape(c) + SUFFIX, neuron)]
        if options:
            assignment[neuron] = [max(options, key=len)]
    assignment = {n: c for n, c in assignment.items() if c}
    unmapped = [n for n in neuron_ids if n not in assignment]
    used = {c for cs in assignment.values() for c in cs}
    return assignment, unmapped, sorted(set(classes) - used), [n for n in AMBIGUOUS if n in assignment]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=str(Path.home() / 'OneDrive/r-elegans'))
    parser.add_argument('--cutoff-nM', type=float, default=500.0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    root = Path(args.data_root)
    beets = root / 'raw/neuropeptides/beets2023'
    prefix = '1-s2.0-S2211124723010690-'
    gpcrs = pd.read_excel(beets / f'{prefix}mmc2.xlsx')
    library = pd.read_excel(beets / f'{prefix}mmc3.xlsx')
    ec50 = pd.read_excel(beets / f'{prefix}mmc6.xlsx')
    neuron_ids = list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
    index = {n: i for i, n in enumerate(neuron_ids)}

    library['genes'] = library['Neuropeptide precursor gene'].ffill().astype(str).map(
        lambda text: sorted(set(re.findall(r'\b([a-z]{2,4}-\d+)\b', re.sub(r'\([^)]*\)', '', text.lower())))))
    peptide_gene = dict(zip(library['Neuropeptide name'], library['genes']))
    gpcrs['gene'] = gpcrs['Gene name'].astype(str).str.lower()
    receptor_names = dict(zip(gpcrs['GPCR ID'], zip(gpcrs['gene'], gpcrs['Gene sequence (WS240)'].astype(str))))
    ec50['precursor'] = ec50['Peptide'].map(peptide_gene)
    ec50.loc[ec50['precursor'].map(lambda g: isinstance(g, list) and not g), 'precursor'] = None
    ec50['receptor_gene'] = ec50['GPCR ID'].map(lambda g: receptor_names.get(g, (None, None))[0])
    ec50['receptor_seq'] = ec50['GPCR ID'].map(lambda g: receptor_names.get(g, (None, None))[1])
    ec50['EC50_nM'] = ec50['EC50 (M)'].astype(float) * 1e9
    summary = dict(ec50_rows=len(ec50), peptides_without_precursor=sorted(ec50.loc[ec50['precursor'].isna(), 'Peptide'].unique().tolist()),
                   receptors_unmapped=sorted(ec50.loc[ec50['receptor_gene'].isna(), 'GPCR ID'].unique().tolist()))
    use = ec50.dropna(subset=['precursor', 'receptor_gene'])
    use = use[use['EC50_nM'] <= args.cutoff_nM]
    summary['pairs_within_cutoff'] = len(use)

    out = Path(args.out) if args.out else root / 'processed/extrasynaptic_candidates_v1'
    out.mkdir(parents=True, exist_ok=True)
    summary['thresholds'] = {}
    for threshold in (1, 2, 3, 4):
        path = {1: 'liberal_threshold1', 2: 'medium_threshold2', 3: 'conservative_threshold3', 4: 'stringent_threshold4'}[threshold]
        table = pd.read_csv(root / f'raw/expression/cengen/021821_{path}.csv')
        classes = [c for c in table.columns if c not in ('Unnamed: 0', '', 'gene_name', 'Wormbase_ID')]
        assignment, unmapped, unused, ambiguous = class_assignment(neuron_ids, classes)
        expressed = {}
        for gene, row in table.set_index(table['gene_name'].astype(str).str.lower())[classes].iterrows():
            expressed[gene] = row
        def neurons_for(gene, alt=None):
            row = expressed.get(gene)
            if row is None and alt:
                row = expressed.get(str(alt).lower())
            if row is None:
                return None
            return np.array([any(row.get(c, 0) > 0 for c in assignment[n]) if n in assignment else False for n in neuron_ids])
        count = np.zeros((len(neuron_ids), len(neuron_ids)), dtype=np.int32)  # [receiver, sender]
        best = np.full(count.shape, np.inf)
        missing_precursors, missing_receptors = set(), set()
        for record in use.itertuples():
            sender_sets = [neurons_for(g) for g in record.precursor]
            for gene, found in zip(record.precursor, sender_sets):
                if found is None:
                    missing_precursors.add(gene)
            sender_sets = [x for x in sender_sets if x is not None]
            senders = np.any(sender_sets, axis=0) if sender_sets else None
            receivers = neurons_for(record.receptor_gene, record.receptor_seq)
            if receivers is None:
                missing_receptors.add(record.receptor_gene)
            if senders is None or receivers is None:
                continue
            mask = np.outer(receivers, senders)
            count[mask] += 1
            best[mask] = np.minimum(best[mask], record.EC50_nM)
        rows = np.argwhere(count > 0)
        frame = pd.DataFrame(dict(sender=[neuron_ids[s] for _, s in rows], receiver=[neuron_ids[r] for r, _ in rows],
                                  n_pairs=count[count > 0].astype(int), min_EC50_nM=best[count > 0]))
        frame.to_csv(out / f'candidate_pairs_cengen{threshold}_ec50le{int(args.cutoff_nM)}nM.csv.gz', index=False)
        summary['thresholds'][threshold] = dict(
            candidate_neuron_pairs=int(len(frame)), senders=int(frame['sender'].nunique()), receivers=int(frame['receiver'].nunique()),
            self_pairs=int((frame['sender'] == frame['receiver']).sum()), median_pairs_per_edge=float(frame['n_pairs'].median()) if len(frame) else 0.0,
            neurons_without_class=unmapped, cengen_classes_unused=unused, ambiguous_neurons=ambiguous,
            precursor_genes_missing_in_cengen=sorted(missing_precursors), receptor_genes_missing_in_cengen=sorted(missing_receptors))
        print('threshold', threshold, {k: v for k, v in summary['thresholds'][threshold].items() if not isinstance(v, list)}, flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('unmapped neurons (t2):', summary['thresholds'][2]['neurons_without_class'])
    print('unused classes (t2):', summary['thresholds'][2]['cengen_classes_unused'])
    print('missing precursors (t2):', summary['thresholds'][2]['precursor_genes_missing_in_cengen'])
    print('missing receptors (t2):', summary['thresholds'][2]['receptor_genes_missing_in_cengen'])
    print({k: v for k, v in summary.items() if k != 'thresholds'})


if __name__ == '__main__':
    main()
