"""Build the bundled peptide-receptor pair asset for the extrasynaptic module (data processing only; no model change).

Each pair k is one Beets et al. 2023 peptide-GPCR combination with EC50 <= cutoff. `sender[k]` marks model neurons whose
CeNGEN class expresses any precursor gene of the peptide, `receiver[k]` those that express the receptor gene (CeNGEN
threshold `--threshold`). No sign, strength, range or time constant is inferred here.
"""
import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from r_elegans.assets import load_asset_document
from scripts.build_peptide_candidates import class_assignment

warnings.filterwarnings('ignore')
THRESHOLD_FILES = {1: 'liberal_threshold1', 2: 'medium_threshold2', 3: 'conservative_threshold3', 4: 'stringent_threshold4'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path.home() / 'OneDrive/r-elegans')
    parser.add_argument('--threshold', type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument('--cutoff-nM', type=float, default=500.)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'r_elegans/assets')
    args = parser.parse_args()
    root = args.data_root
    beets = root / 'raw/neuropeptides/beets2023'
    prefix = '1-s2.0-S2211124723010690-'
    gpcrs = pd.read_excel(beets / f'{prefix}mmc2.xlsx')
    library = pd.read_excel(beets / f'{prefix}mmc3.xlsx')
    ec50 = pd.read_excel(beets / f'{prefix}mmc6.xlsx')
    cengen = root / f'raw/expression/cengen/021821_{THRESHOLD_FILES[args.threshold]}.csv'
    neuron_ids = list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
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
    use = ec50.dropna(subset=['precursor', 'receptor_gene'])
    use = use[use['EC50_nM'] <= args.cutoff_nM]

    table = pd.read_csv(cengen)
    classes = [c for c in table.columns if c not in ('Unnamed: 0', '', 'gene_name', 'Wormbase_ID')]
    assignment, unmapped, _, _ = class_assignment(neuron_ids, classes)
    expressed = {gene: row for gene, row in table.set_index(table['gene_name'].astype(str).str.lower())[classes].iterrows()}

    def neurons_for(gene, alt=None):
        row = expressed.get(gene)
        if row is None and alt:
            row = expressed.get(str(alt).lower())
        if row is None:
            return None
        return np.array([any(row.get(c, 0) > 0 for c in assignment[n]) if n in assignment else False for n in neuron_ids])

    senders, receivers, strengths, labels, skipped = [], [], [], [], []
    for record in use.itertuples():
        sender_sets = [x for x in (neurons_for(g) for g in record.precursor) if x is not None]
        sender = np.any(sender_sets, axis=0) if sender_sets else None
        receiver = neurons_for(record.receptor_gene, record.receptor_seq)
        if sender is None or receiver is None or not sender.any() or not receiver.any():
            skipped.append(dict(peptide=record.Peptide, receptor=record.receptor_gene))
            continue
        senders.append(sender)
        receivers.append(receiver)
        strengths.append(record.EC50_nM)
        labels.append(dict(peptide=record.Peptide, precursor=list(record.precursor), receptor=record.receptor_gene))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'extrasynaptic_v1.npz', sender=np.asarray(senders, np.uint8), receiver=np.asarray(receivers, np.uint8),
                        ec50_nM=np.asarray(strengths, float), neuron_ids=np.asarray(neuron_ids))
    metadata = dict(
        version='extrasynaptic_v1', cengen_threshold=args.threshold, cutoff_nM=args.cutoff_nM, pairs=len(senders),
        pairs_skipped_without_expression=skipped, neurons_without_cengen_class=unmapped,
        source_sha256={p.name: sha256(p) for p in (beets / f'{prefix}mmc2.xlsx', beets / f'{prefix}mmc3.xlsx', beets / f'{prefix}mmc6.xlsx', cengen)},
        pair_labels=labels,
        sender_counts_per_pair=dict(min=int(np.min(np.sum(senders, axis=1))), median=float(np.median(np.sum(senders, axis=1))), max=int(np.max(np.sum(senders, axis=1)))),
        receiver_counts_per_pair=dict(min=int(np.min(np.sum(receivers, axis=1))), median=float(np.median(np.sum(receivers, axis=1))), max=int(np.max(np.sum(receivers, axis=1)))),
        note='Candidate peptide-receptor pairs from expression x EC50 only; no sign, strength, range or time constant inferred.')
    (out / 'extrasynaptic_v1.json').write_text(json.dumps(metadata, indent=1) + '\n')
    print(json.dumps({k: v for k, v in metadata.items() if k not in ('pair_labels', 'source_sha256')}, default=str)[:1500])


if __name__ == '__main__':
    main()
