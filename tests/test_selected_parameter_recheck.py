import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiment_records import file_sha256
from scripts.recheck_selected_parameters import recheck
from scripts.selected_atlas_audit_checks import parameter_audit

BUNDLE = Path('r_elegans/assets/mac_rhythm_v1_polarized/network.npz')


@pytest.mark.parametrize('key,bound', [('cells_capacitance_pF', 20.), ('fixed_release_slope_mV', 2.)])
def test_existing_tolerance_survives_float32_anchor(tmp_path, key, bound):
    with np.load(BUNDLE, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    assert data[key].dtype == np.float32
    direction = np.inf if bound == 20. else -np.inf
    data[key] = np.full(data[key].shape, np.nextafter(bound, direction), dtype=np.float64)
    path = tmp_path / 'selected.npz'
    np.savez(path, **data)
    audit = parameter_audit(path, BUNDLE, BUNDLE, 'E104', {'parameter_set': 'all'})
    assert audit['passed'], audit['errors']
    assert audit['bounds'][key]['absolute_tolerance'] == 1e-8
    assert audit['bounds'][key]['bound_arithmetic_dtype'] == 'float64'
    data[key][0] = bound + (2e-8 if bound == 20. else -2e-8)
    np.savez(path, **data)
    audit = parameter_audit(path, BUNDLE, BUNDLE, 'E104', {'parameter_set': 'all'})
    assert 'Per-cell bound exceeded: ' + key in audit['errors']


def test_recheck_preserves_original_audit_and_rejects_tampering(tmp_path):
    import shutil
    run = tmp_path / 'audit'
    inputs = run / 'inputs' / 'E104-example'
    inputs.mkdir(parents=True)
    digest = file_sha256(BUNDLE)
    for name in ('network_best_val.npz', 'parent_network.npz', 'anchor_network.npz'):
        shutil.copyfile(BUNDLE, inputs / name)
    (inputs / 'summary.json').write_text(json.dumps(dict(
        experiment='E104', best_val_sha256=digest, parent_sha256=digest, anchor_sha256=digest)))
    manifest = inputs / 'manifest.json'
    manifest.write_text(json.dumps(dict(arguments={'parameter_set': 'all'}, evidence=[
        dict(path=str(p.resolve()), sha256=file_sha256(p)) for p in inputs.iterdir()])))
    (run / 'E104-example-job.json').write_text(json.dumps(dict(
        inputs=str(inputs), manifest=str(manifest), manifest_sha256=file_sha256(manifest))))
    cohort = run / 'summary.json'
    cohort.write_text(json.dumps(dict(audit_complete=True, preparation_failures=[], finished=[dict(
        label='E104-example', result=dict(checkpoint_sha256=digest, checks=dict(parameters=dict(
            passed=False, errors=['historical audit failure']))))])))
    original = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
    output = tmp_path / 'recheck'
    output.mkdir()
    result = recheck(run, output)
    assert result['cases'][0]['corrected_parameters']['passed']
    assert not result['cases'][0]['previous_parameters']['passed']
    assert not result['model_promoted']
    assert original == {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
    with (inputs / 'network_best_val.npz').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='checksum absent or mismatched'):
        recheck(run, output)
