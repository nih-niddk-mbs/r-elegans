"""Summarize and validate the Randi 2023 segmentation files: labeling coverage and target-column consistency."""
import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from r_elegans.assets import load_asset_document
from r_elegans.data.randi_atlas import load_randi_animal, stimulus_responses

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path.home() / 'OneDrive/r-elegans/raw/functional/randi2023_dandi001075_segmentation'))
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    neuron_ids = list(load_asset_document('runtime_model_v1.json')['neuron_ids'])
    rows, match, mismatch = [], Counter(), Counter()
    pair_counts, target_counts, status_total = Counter(), Counter(), Counter()
    for path in sorted(Path(args.root).glob('sub-*/*.nwb')):
        a = load_randi_animal(path, neuron_ids)
        responses = stimulus_responses(a)
        named = [i for i, n in enumerate(a['names']) if n]
        has_target = a['stim_target_column'] >= 0
        top_ok = 0
        checked = 0
        for k in np.flatnonzero(has_target):
            r = responses[k]
            if not np.isfinite(r).any():
                continue
            checked += 1
            col = int(a['stim_target_column'][k])
            best = int(np.nanargmax(r))
            if best == col:
                top_ok += 1
            for offset in (-1, 1):
                if 0 <= col + offset < r.size and best == col + offset:
                    mismatch[offset] += 1
        match['checked'] += checked
        match['target_is_max'] += top_ok
        for k in np.flatnonzero(has_target):
            col = int(a['stim_target_column'][k])
            target = a['names'][col]
            if target:
                target_counts[target] += 1
                for i in named:
                    if i != col and np.isfinite(responses[k, i]):
                        pair_counts[(target, a['names'][i])] += 1
        status_total.update(a['status'])
        rows.append(dict(file=path.name, genotype=a['genotype'], duration_s=float(a['time_s'][-1] - a['time_s'][0]), rois=len(a['names']), named=len(named),
                         stimulations=int(a['stim_start_s'].size), with_target=int(has_target.sum()),
                         with_named_target=int(sum(bool(a['names'][c]) for c in a['stim_target_column'][has_target])),
                         interpolated_nan_fraction=float(np.isnan(a['green']).mean())))
    n = len(rows)
    summary = dict(animals=n, genotypes=dict(Counter(r['genotype'] for r in rows)),
                   rois_total=sum(r['rois'] for r in rows), named_total=sum(r['named'] for r in rows),
                   median_named_per_animal=float(np.median([r['named'] for r in rows])), median_rois=float(np.median([r['rois'] for r in rows])),
                   stimulations=sum(r['stimulations'] for r in rows), with_target=sum(r['with_target'] for r in rows),
                   with_named_target=sum(r['with_named_target'] for r in rows), distinct_named_targets=len(target_counts),
                   distinct_target_responder_pairs=len(pair_counts), roi_status=dict(status_total),
                   target_column_check=dict(checked=match['checked'], target_is_max=match['target_is_max'], max_is_next_column=mismatch[1], max_is_previous_column=mismatch[-1]),
                   median_interpolated_nan_fraction=float(np.median([r['interpolated_nan_fraction'] for r in rows])))
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(dict(summary=summary, animals=rows), indent=1))


if __name__ == '__main__':
    main()
