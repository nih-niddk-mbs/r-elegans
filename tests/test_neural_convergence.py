"""E107 protocol/audit tests use synthetic saved data; no scientific run."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiment_records import file_sha256
from scripts.run_neural_convergence import ATLAS_STEPS, RHYTHM_STEPS_MS, audit, prepare

BUNDLE = Path('r_elegans/assets/mac_rhythm_v1_polarized/network.npz')


def fixture_study(root):
    root.mkdir()
    digest = file_sha256(BUNDLE)
    manifest = dict(evidence_sha256={str(BUNDLE): digest}, models={
        n: dict(path=str(BUNDLE), sha256=digest) for n in ('selected', 'parent')})
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    time = np.arange(20000) * .002
    voltage = np.stack([10 * np.sin(2 * np.pi * time / .25), 10 * np.cos(2 * np.pi * time / .25)], axis=1)
    atlas = np.arange(14 * 302, dtype=float).reshape(14, 302) / 1e6
    atlas[-1] = 0
    for label in manifest['models']:
        folder = root / label
        folder.mkdir()
        levels = []
        for i, (substeps, step_ms) in enumerate(zip(ATLAS_STEPS, RHYTHM_STEPS_MS)):
            artifact = folder / ('level-' + str(i) + '.npz')
            np.savez_compressed(artifact, atlas_response=atlas, rmd_voltage_mV=voltage,
                                all_cell_late_std_mV=np.ones(302), sample_dt_s=.002,
                                atlas_step_ms=100 / substeps, rhythm_step_ms=step_ms)
            levels.append(dict(level=i, artifact=str(artifact), sha256=file_sha256(artifact)))
        (folder / 'summary.json').write_text(json.dumps(dict(levels=levels, checkpoint_sha256=digest,
                                                           manifest_sha256=file_sha256(path))))
    return root


def rewrite_level(root, index, update):
    summary_path = root / 'selected' / 'summary.json'
    summary = json.loads(summary_path.read_text())
    item = summary['levels'][index]
    path = Path(item['artifact'])
    with np.load(path) as z:
        arrays = dict(z)
    update(arrays)
    np.savez_compressed(path, **arrays)
    item['sha256'] = file_sha256(path)
    summary_path.write_text(json.dumps(summary))


def test_complete_hash_checked_two_pair_convergence(tmp_path):
    root = fixture_study(tmp_path / 'study')
    result = audit(root, complete_execution=True)
    assert result['integrity_passed'], result['problems']
    assert result['numerical_component_passed']
    assert not result['model_promoted']
    assert not result['swimming_qualified']
    assert len(result['models']['selected']['comparisons']) == 4


@pytest.mark.parametrize('failure', ['earlier_fine_pair', 'rhythm_lost', 'sham', 'metadata', 'nonfinite', 'tamper', 'missing'])
def test_negative_saved_evidence_never_qualifies(tmp_path, failure):
    root = fixture_study(tmp_path / 'study')
    if failure == 'earlier_fine_pair':
        # Last pair passes but penultimate pair fails: one good comparison is insufficient.
        rewrite_level(root, 2, lambda a: a.__setitem__('atlas_response', a['atlas_response'] * 2))
    elif failure == 'rhythm_lost':
        rewrite_level(root, 4, lambda a: a.__setitem__('rmd_voltage_mV', np.zeros((20000, 2))))
    elif failure == 'sham':
        rewrite_level(root, 4, lambda a: a['atlas_response'][-1].fill(1e-5))
    elif failure == 'metadata':
        rewrite_level(root, 4, lambda a: a.__setitem__('rhythm_step_ms', np.array(99.)))
    elif failure == 'nonfinite':
        rewrite_level(root, 4, lambda a: a['atlas_response'][0].fill(np.nan))
    elif failure == 'tamper':
        with (root / 'selected' / 'level-4.npz').open('ab') as stream:
            stream.write(b'changed')
    else:
        p = root / 'selected' / 'summary.json'
        s = json.loads(p.read_text());s['levels'].pop();p.write_text(json.dumps(s))
    result = audit(root, complete_execution=True)
    assert not result['numerical_component_passed']
    assert not result['models']['selected']['numerical_component_passed']


def test_timeout_cannot_qualify_even_with_complete_arrays(tmp_path):
    result = audit(fixture_study(tmp_path / 'study'), complete_execution=False)
    assert result['integrity_passed']
    assert not result['numerical_component_passed']


def test_prepare_rejects_unverified_manifest_before_forward_work(tmp_path):
    source = tmp_path / 'source';source.mkdir()
    manifest = source / 'manifest.json';manifest.write_text('{}')
    (source / 'E104-network_control-job.json').write_text(json.dumps(dict(
        manifest=str(manifest), manifest_sha256='wrong')))
    with pytest.raises(ValueError, match='manifest hash mismatch'):
        prepare(source, tmp_path / 'output')


def test_prepare_copies_pinned_inputs_and_preserves_source(tmp_path, monkeypatch):
    import shutil
    import scripts.run_neural_convergence as study
    source = tmp_path / 'source';source.mkdir()
    inputs = source / 'inputs';inputs.mkdir()
    digest = file_sha256(BUNDLE)
    monkeypatch.setattr(study, 'SELECTED_SHA', digest)
    monkeypatch.setattr(study, 'PARENT_SHA', digest)
    for name in ('network_best_val.npz', 'parent_network.npz'):
        shutil.copyfile(BUNDLE, inputs / name)
    targets = ['target-' + str(i) for i in range(13)]
    np.savez(inputs / 'responses.npz', targets=targets, stimulus_nS=.5)
    (inputs / 'summary.json').write_text(json.dumps(dict(best_val_sha256=digest, parent_sha256=digest)))
    manifest = inputs / 'manifest.json'
    manifest.write_text(json.dumps(dict(arguments=dict(stimulus_nS=.5), statistics=dict(targets=targets),
        evidence=[dict(path=str(p.resolve()), sha256=file_sha256(p)) for p in inputs.iterdir()])))
    (source / 'E104-network_control-job.json').write_text(json.dumps(dict(
        manifest=str(manifest), manifest_sha256=file_sha256(manifest), inputs=str(inputs))))
    original = {p: file_sha256(p) for p in source.rglob('*') if p.is_file()}
    out = tmp_path / 'out';out.mkdir()
    result = prepare(source, out)
    assert result['targets'] == targets
    assert all(file_sha256(m['path']) == digest for m in result['models'].values())
    assert original == {p: file_sha256(p) for p in source.rglob('*') if p.is_file()}


def test_supervisor_timeout_runs_audit_and_records_failure(tmp_path, monkeypatch):
    import scripts.run_neural_convergence as study
    import scripts.total_job_budget as budget
    root = fixture_study(tmp_path / 'template')
    output = tmp_path / 'output'
    def prepare_mock(source, out):
        import shutil
        for item in root.iterdir():
            target = out / item.name
            if item.is_dir():
                shutil.copytree(item, target)
                summary = target / 'summary.json'
                s = json.loads(summary.read_text())
                for row in s['levels']:
                    row['artifact'] = str(target / Path(row['artifact']).name)
                summary.write_text(json.dumps(s))
            else:
                shutil.copyfile(item, target)
        return dict(targets=['synthetic'])
    monkeypatch.setattr(study, 'capture_run_record', lambda *a, **kw: None)
    monkeypatch.setattr(study, 'prepare', prepare_mock)
    def timeout(command, seconds):
        assert 0 < seconds < 240  # five minutes less the reserved audit minute
        return 124, True
    monkeypatch.setattr(budget, 'run_group', timeout)
    monkeypatch.setattr('sys.argv', ['study', '--source-audit', str(tmp_path / 'source'),
                                   '--output-dir', str(output), '--minutes', '5'])
    assert study.main() == 124
    receipt = json.loads((output / 'budget_receipt.json').read_text())
    assert receipt['audit_complete']
    assert receipt['timed_out']
    assert not receipt['all_steps_finished']
    assert not receipt['numerical_component_passed']
