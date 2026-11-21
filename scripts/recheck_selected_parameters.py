"""Recheck E106 saved parameter bounds with float64 bound arithmetic; no simulation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from scripts.experiment_records import capture_run_record, file_sha256
from scripts.selected_atlas_audit_checks import parameter_audit, saved_arrays, write


def recheck(run, output):
    """Preserve old audit files and verify every input consumed by this recheck."""
    cohort_path = run / 'summary.json'
    cohort = json.loads(cohort_path.read_text())
    if not cohort.get('audit_complete') or cohort.get('preparation_failures'):
        raise ValueError('Requires a completed cohort without preparation failures')
    evidence = {str(cohort_path): file_sha256(cohort_path)}
    rows = []
    for row in cohort['finished']:
        label = row['label']
        job_path = run / (label + '-job.json')
        evidence[str(job_path)] = file_sha256(job_path)
        job = json.loads(job_path.read_text())
        manifest_path = Path(job['manifest'])
        actual = file_sha256(manifest_path)
        if actual != job['manifest_sha256']:
            raise ValueError('Manifest changed: ' + label)
        evidence[str(manifest_path)] = actual
        manifest = json.loads(manifest_path.read_text())
        declared = {str(Path(e['path']).resolve()): e['sha256'] for e in manifest['evidence']}
        inputs = Path(job['inputs'])

        def verify(name):
            path = inputs / name
            actual = file_sha256(path)
            if declared.get(str(path.resolve())) != actual:
                raise ValueError('Input checksum absent or mismatched: ' + str(path))
            evidence[str(path)] = actual
            return path

        selected, parent, anchor, summary = map(verify, (
            'network_best_val.npz', 'parent_network.npz', 'anchor_network.npz', 'summary.json'))
        s = json.loads(summary.read_text())
        for name, path in (('best_val_sha256', selected), ('parent_sha256', parent), ('anchor_sha256', anchor)):
            if evidence[str(path)] != s[name]:
                raise ValueError('Source summary checksum mismatch: ' + name)
        if evidence[str(selected)] != row['result']['checkpoint_sha256']:
            raise ValueError('Recheck checkpoint differs from original audit: ' + label)
        audit = parameter_audit(selected, parent, anchor, s['experiment'], manifest['arguments'])
        masks = audit.pop('masks', {})
        if (inputs / 'trainable_masks.npz').exists():
            recorded = saved_arrays(verify('trainable_masks.npz'))
            mapping = {'channels': 'params_log_channel_scale', 'chemical': 'params_raw_chemical', 'gap': 'params_raw_gap'}
            audit['recorded_masks_match'] = all(np.array_equal(recorded[k], masks[v]) for k, v in mapping.items())
            if not audit['recorded_masks_match']:
                audit['passed'] = False
                audit['errors'].append('Recorded masks differ from declared policy')
        rows.append(dict(label=label, checkpoint_sha256=evidence[str(selected)],
                         previous_parameters=row['result']['checks']['parameters'], corrected_parameters=audit))
    for path, expected in evidence.items():
        if file_sha256(path) != expected:
            raise ValueError('Input changed during recheck: ' + path)
    result = dict(source_run=str(run), recheck_complete=True, cases=rows,
                  evidence_sha256=evidence, model_promoted=False,
                  scope='Parameter arithmetic correction only; original audit unchanged. No numerical, rhythm, performance or behavior gates reclassified.')
    write(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output_dir or run.with_name(run.name + '-parameter-recheck-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    capture_run_record(output, Path(__file__).resolve().parents[1], args, phase='saved-parameter-recheck')
    result = recheck(run, output)
    for row in result['cases']:
        audit = row['corrected_parameters']
        print(row['label'], dict(previous_errors=row['previous_parameters'].get('errors'),
                                 corrected_errors=audit['errors'], parameter_passed=audit['passed']))
    print('Report:', output / 'summary.json')
    print(result['scope'])


if __name__ == '__main__':
    main()
