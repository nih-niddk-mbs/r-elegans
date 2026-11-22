import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiment_records import file_sha256
from scripts.run_common_state import audit, STEPS, FIELDS

BUNDLE = Path('r_elegans/assets/mac_rhythm_v1_polarized/network.npz')


def test_reference_phases_require_a_resolved_cycle():
    from r_elegans.brain.common_state_replay import phase_indices
    t = (np.arange(1000) + 1) * .001
    v = 10 * np.sin(2 * np.pi * t / .25)
    ix, peaks = phase_indices(v, .001)
    assert len(set(ix)) == 4
    np.testing.assert_array_equal(ix, np.rint(peaks[0] + np.arange(4) * (peaks[1] - peaks[0]) / 4).astype(int))
    with pytest.raises(ValueError, match='No complete'):
        phase_indices(np.zeros(1000), .001)


def test_replay_preserves_complete_state_and_matches_direct_short_step():
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.common_state_replay import replay
    from r_elegans.brain.conductance_network import load_network, initial_network_state, advance_network, MembraneConductanceInput
    p, f, c = load_network(BUNDLE)
    p, f = jax.tree.map(lambda a: a.astype(jnp.float64) if jnp.issubdtype(a.dtype, jnp.floating) else a, (p, f))
    state = initial_network_state(f, c)
    anchors = jax.tree.map(lambda a: jnp.stack([a, a]), state)
    before = jax.tree.map(np.asarray, anchors)
    g = jnp.zeros((3, 302)).at[0, 0].set(.5)
    means, short, voltage = replay(p, f, c, anchors, g, [0, 1], .25,
        seconds=.012, pulse_seconds=.003, short_seconds=.002, window=(.004, .01))
    direct = state
    for _ in range(2):
        direct = advance_network(direct, p, f, .001, 4, c,
            membrane_input=MembraneConductanceInput(g[0], jnp.zeros(302)))
    for key in FIELDS:
        np.testing.assert_allclose(getattr(short, key)[0, 0], getattr(direct, key), atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(getattr(anchors, key), getattr(before, key))
        np.testing.assert_array_equal(getattr(short, key)[:, 1], getattr(short, key)[:, 2])
    np.testing.assert_array_equal(means[:, 1], means[:, 2])
    np.testing.assert_array_equal(voltage[:, 1], voltage[:, 2])
    assert np.max(abs(voltage[0, 0] - voltage[0, 1])) > 1e-6


def fixture_job(root, monkeypatch):
    import scripts.run_common_state as job
    from r_elegans.brain.common_state_replay import phase_indices
    root.mkdir()
    digest = file_sha256(BUNDLE)
    monkeypatch.setattr(job, 'SELECTED_SHA', digest);monkeypatch.setattr(job, 'PARENT_SHA', digest)
    with np.load(BUNDLE) as z:
        ids = z['neuron_ids'].tolist()
    gs = np.zeros((5, 302));gs[np.arange(3), [ids.index(n) for n in job.TARGETS]] = .5
    manifest = dict(models={n:dict(path=str(BUNDLE), sha256=digest) for n in ('selected', 'parent')},
                    evidence_sha256={str(BUNDLE):digest}, stimulus_nS=.5, source_numerical_component_passed=False)
    (root / 'manifest.json').write_text(json.dumps(manifest));manifest_sha = file_sha256(root / 'manifest.json')
    t = (np.arange(1000) + 1) * .001
    reference = np.stack([10 * np.sin(2 * np.pi * t / .25), 10 * np.cos(2 * np.pi * t / .25)], axis=1)
    ix, peaks = phase_indices(reference[:, 0], .001)
    starts = {k:np.zeros((4, 302, 35) if k == 'gates' else (4, 302), dtype=float) for k in FIELDS}
    starts['voltage_mV'][:, [ids.index('RMDL'), ids.index('RMDR')]] = reference[ix]
    tlong = (np.arange(12000) + 1) * .001
    v = np.stack([10 * np.sin(2 * np.pi * tlong / .25), 10 * np.cos(2 * np.pi * tlong / .25)], axis=1)
    voltage = np.broadcast_to(v, (4, 5, 12000, 2))
    means = np.broadcast_to(np.arange(302, dtype=float) * 1e-6, (4, 5, 302)).copy()
    means[:, :3] *= 2
    for label in ('selected', 'parent'):
        folder = root / label;folder.mkdir()
        path = folder / 'anchors.npz'
        np.savez_compressed(path, **{'start_' + k:v for k,v in starts.items()}, conductances_nS=gs,
                            reference_rmd_mV=reference, anchor_indices=ix, reference_peak_indices=peaks)
        anchor_sha = file_sha256(path);levels = []
        for i, step in enumerate(STEPS):
            path = folder / ('level-' + str(i) + '.npz')
            np.savez_compressed(path, indicator_means=means, rmd_voltage_mV=voltage,
                **{'short_' + k:np.zeros((4, 5, 302, 35) if k == 'gates' else (4, 5, 302)) for k in FIELDS},
                anchor_sha256=anchor_sha, neural_step_ms=step, sample_dt_s=.001)
            levels.append(dict(level=i, artifact=str(path), sha256=file_sha256(path)))
        (folder / 'summary.json').write_text(json.dumps(dict(checkpoint_sha256=digest,
            manifest_sha256=manifest_sha, anchor_sha256=anchor_sha, levels=levels,
            anchor_reference=dict(indices=ix.tolist(), peak_indices=peaks, time_s=((ix + 1) * .001).tolist(),
                                  sample_dt_s=.001, reference_step_ms=.03125, warmup_seconds=60.))))
    return root


def test_saved_audit_complete_and_no_promotion(tmp_path, monkeypatch):
    root = fixture_job(tmp_path / 'job', monkeypatch)
    result = audit(root, True)
    assert result['integrity_passed'], result['problems']
    assert all(m['per_phase_response_checks_passed'] and m['phase_mean_response_checks_passed'] for m in result['models'].values())
    assert not result['model_promoted']
    assert not result['source_numerical_component_passed']
    incomplete = audit(root, False)
    assert not any(m['per_phase_response_checks_passed'] for m in incomplete['models'].values())


@pytest.mark.parametrize('failure', ['tamper', 'anchor_identity', 'sham', 'missing'])
def test_failed_evidence_cannot_pass(tmp_path, monkeypatch, failure):
    root = fixture_job(tmp_path / 'job', monkeypatch)
    summary_path = root / 'selected' / 'summary.json'
    summary = json.loads(summary_path.read_text())
    path = root / 'selected' / 'level-3.npz'
    if failure == 'tamper':
        with path.open('ab') as f:f.write(b'changed')
    elif failure == 'missing':
        summary['levels'].pop()
    else:
        with np.load(path) as z:data = dict(z)
        if failure == 'anchor_identity':data['anchor_sha256'] = np.asarray('wrong')
        else:data['indicator_means'][:, 4] += .001
        np.savez_compressed(path, **data);summary['levels'][3]['sha256'] = file_sha256(path)
    summary_path.write_text(json.dumps(summary))
    result = audit(root, True)
    assert not result['models']['selected'].get('per_phase_response_checks_passed')
    assert not result['models']['selected'].get('phase_mean_response_checks_passed')


def test_preparation_pins_source_copies_and_rejects_changes(tmp_path, monkeypatch):
    import shutil
    import scripts.run_common_state as job
    digest = file_sha256(BUNDLE)
    monkeypatch.setattr(job, 'SELECTED_SHA', digest);monkeypatch.setattr(job, 'PARENT_SHA', digest)
    source = tmp_path / 'e107';source.mkdir()
    models = {}
    for label in ('selected', 'parent'):
        path = source / (label + '.npz');shutil.copyfile(BUNDLE, path)
        models[label] = dict(path=str(path), sha256=digest)
    (source / 'manifest.json').write_text(json.dumps(dict(models=models, targets=list(job.TARGETS),
        stimulus_nS=.5, evidence_sha256={v['path']:digest for v in models.values()})))
    (source / 'audit.json').write_text(json.dumps(dict(audit_complete=True, integrity_passed=True,
                                                       numerical_component_passed=False)))
    (source / 'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True)))
    out = tmp_path / 'e108';out.mkdir()
    result = job.prepare(source, out)
    assert not result['source_numerical_component_passed']
    assert all(file_sha256(v['path']) == digest for v in result['models'].values())
    with Path(models['selected']['path']).open('ab') as stream:stream.write(b'changed')
    next_out = tmp_path / 'bad';next_out.mkdir()
    with pytest.raises(ValueError, match='hash mismatch'):
        job.prepare(source, next_out)


def test_supervisor_timeout_still_audits_without_promotion(tmp_path, monkeypatch):
    import scripts.run_common_state as job
    import scripts.total_job_budget as budget
    monkeypatch.setattr(job, 'capture_run_record', lambda *a, **kw: None)
    def prepared(source, root):
        (root / 'manifest.json').write_text(json.dumps(dict(evidence_sha256={}, source_numerical_component_passed=False,
                                                           models={})))
    monkeypatch.setattr(job, 'prepare', prepared)
    def timeout(command, seconds):
        assert 0 < seconds < 240
        return 124, True
    monkeypatch.setattr(budget, 'run_group', timeout)
    root = tmp_path / 'output'
    monkeypatch.setattr('sys.argv', ['job', '--source-run', str(tmp_path / 'source'),
                                   '--output-dir', str(root), '--minutes', '5'])
    assert job.main() == 124
    receipt = json.loads((root / 'budget_receipt.json').read_text())
    assert receipt['audit_complete'] and receipt['timed_out']
    assert not receipt['all_steps_finished'] and not receipt['model_promoted']
    assert json.loads((root / 'audit.json').read_text())['problems']
