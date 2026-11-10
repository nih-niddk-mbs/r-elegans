"""Prepare external paired targets from an already checksum-verified extraction.

Run as python -m scripts.prepare_paired_recordings. This is data preparation,
not a joint fitter. The final manifest explicitly retains admission blockers.
"""
import argparse
import bz2
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from r_elegans.data.paths import get_data_root
from r_elegans.data.paired_recordings import grouped_splits, load_paired
from scripts.experiment_records import capture_run_record


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = get_data_root(args.output_dir, must_exist=False)
    out.mkdir(parents=True, exist_ok=False)  # immutable result; never overwrite a split
    capture_run_record(out, Path(__file__).resolve().parents[1], args)
    rows = list(csv.DictReader(args.metadata.open()))
    if len({r['uid'] for r in rows}) != len(rows):
        raise ValueError('Duplicate recording UID')
    folds = grouped_splits(rows)
    split_doc = dict(schema_version=1, algorithm='E049-v1 SHA256 date groups, 20% each heldout',
                     grouping='Recording date; animal independence not verified',
                     previously_fitted_dates=['2022-06-14'], assignments=folds)
    (out/'splits.json').write_text(json.dumps(split_doc, indent=2, sort_keys=True)+'\n')
    labels = json.loads(bz2.decompress(args.labels.read_bytes()))['data']
    asset = Path(__file__).resolve().parents[1]/'r_elegans/assets/runtime_model_v1.json'
    neuron_ids = json.loads(asset.read_text())['neuron_ids']
    records = []
    for row in rows:
        name = row['filename']
        if Path(name).name != name:
            raise ValueError('Unsafe metadata filename')
        path = args.recordings/name
        digest = sha(path)
        if digest != row['checksum']:
            raise ValueError('Publisher SHA256 mismatch: '+name)
        record = dict(uid=row['uid'], date_group=row['uid'][:10], fold=folds[row['uid']],
                      source_sha256=digest, type=row['type'], event=row['event'])
        if row['type'] != 'baseline,neuropal':
            record.update(status='quarantined', reason='Not identified baseline; thermal/GFP/unlabeled conditions outside initial cohort')
        else:
            if row['θh_pos_is_ventral'] not in ('true', 'false'):
                raise ValueError('Unknown orientation')
            arrays, qc = load_paired(path, labels[row['uid']]['idx_neuron-label'], neuron_ids,
                                      ventral=row['θh_pos_is_ventral']=='true')
            target = out/(row['uid']+'.npz')
            np.savez_compressed(target, **arrays)
            record.update(status='prepared_not_fit_admitted', qc=qc,
                          targets_sha256=sha(target), targets_file=target.name)
        records.append(record)
        print(json.dumps({k: record[k] for k in ('uid','fold','status')}), flush=True)
    summary = dict(schema_version=1, experiment='E049', status='prepared_with_explicit_admission_blockers',
                   metadata_sha256=sha(args.metadata), labels_sha256=sha(args.labels),
                   split_sha256=sha(out/'splits.json'), canonical_model_asset_sha256=sha(asset),
                   normalization='None. Raw author-processed ratiometric traces; no held-out calibration fit.',
                   alignment='Publisher confocal-time behavior arrays; no added interpolation',
                   fluorescence_unit='Author-processed ratio; not absolute calcium concentration or voltage',
                   limitations=[
                       'Published cohort/mounting protocol recorded; exact per-UID genotype and recording temperature missing; coverslip/2% agar contact mechanics not calibrated',
                       'Verify reuse license separately; downloaded publisher metadata does not specify it',
                       'Publisher uses within-record interpolation and full-record bleach estimation; finite masks do not identify these imputed samples; no added cross-fold normalization',
                       'No full-body arclength/orientation conversion to model joints; source angles retained only',
                       'No calibrated calcium indicator likelihood or training-only normalization fitted',
                       'Nonidentity ROI index spaces quarantined until verified',
                       'Date-group folds do not establish animal-level independence; not a prospectively unseen dataset'],
                   records=records)
    summary['protocol'] = json.loads((Path(__file__).resolve().parents[1]/
        'experiments/paired_recordings_protocol_v1.json').read_text())
    (out/'manifest.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    print('Manifest:', out/'manifest.json', flush=True)


if __name__ == '__main__':
    main()
