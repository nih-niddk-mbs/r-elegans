import json
import sys
import numpy as np
import pytest
from scripts import run_full_cycle_validation as run


def manifest(targets=None):
    targets = targets or ['A', 'B', 'C', 'D', 'E']
    return dict(panel=targets, source_panel=targets[:3],
                target_batches=[dict(begin=0, targets=targets[:3]), dict(begin=3, targets=targets[3:])],
                cases=[dict(arm='peptide', level=n, model_sha256='fixed', tolerance=1e-5,
                            base_cycle_times=[60., 61., 62., 63.], extension_start_s=64.) for n in (0, 1)],
                numeric_tolerance=1e-5)


def test_panel_batch_is_reusable_and_full_targets_are_covered_once():
    m = manifest(['T'+str(i) for i in range(135)])
    m['target_batches'] = [dict(begin=0, targets=m['panel'][:3])] + [dict(begin=i, targets=m['panel'][i:i+4]) for i in range(3, 135, 4)]
    tasks = run.tasks_for(m)
    assert len(tasks) == 476
    assert [t for t in tasks if t['level'] == 0 and t['group'] == 0][0]['targets'] == ['T0', 'T1', 'T2']
    for level in (0, 1):
        for group in range(7):
            names = sum([t['targets'] for t in tasks if t['level'] == level and t['group'] == group], [])
            assert names == m['panel']


def test_validation_finishes_all_batches_even_after_planning_allowance(tmp_path, monkeypatch):
    m = manifest()
    monkeypatch.setattr(run.time, 'time', lambda: 10**20)
    stream = run.task_iterator(tmp_path, m, 0)
    tasks = list(stream)
    assert tasks == run.tasks_for(m)[::2]
    assert json.loads((tmp_path/'admitted-0.json').read_text()) == tasks[-1]
    # Crash recovery repeats no completed batch; interrupted kernels may restart.
    save_batch(tmp_path, tasks[0])
    assert list(run.task_iterator(tmp_path, m, 0)) == tasks[1:]


def test_readonly_actions_never_start_or_create_work(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'BASE', tmp_path)
    monkeypatch.setattr(run, 'jobs', lambda: [])
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **k: pytest.fail('Unexpected launch'))
    for args in (['--status'], ['--select-output'], ['--output-dir', str(tmp_path/'absent'), '--report']):
        monkeypatch.setattr(sys, 'argv', ['full']+args)
        assert run.main() == 0
    assert list(tmp_path.iterdir()) == []


def test_active_work_refused_before_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'jobs', lambda: [dict(pid=123)])
    monkeypatch.setattr(sys, 'argv', ['full', '--output-dir', str(tmp_path/'new')])
    with pytest.raises(SystemExit):
        run.main()
    assert not (tmp_path/'new').exists()


def save_batch(root, task, value=.001):
    means = np.zeros((16, 6, 302)); means[:, :len(task['targets'])] = value
    t = np.arange(12000)*.001
    sham = np.broadcast_to((-30+10*np.sin(2*np.pi*t*4))[None, None, :, None], (16, 2, 12000, 2)).copy()
    data = dict(means=means, voltage_min=np.full((16, 6), -40.), voltage_max=np.full((16, 6), -20.),
                sham_voltage=sham if task['begin'] == 0 else np.empty(0), final_gates=np.zeros((16, 6, 302, 35)))
    for key in ('voltage_mV', 'calcium_uM', 'indicator', 'peptide'):
        data['final_'+key] = np.zeros((16, 6, 302))
    run.resume.atomic_npz(run.cycle.task_path(root, task), provenance_json=np.array(json.dumps(task)), **data)


def test_partial_coverage_is_not_full_qualification_then_full_mean_passes(tmp_path):
    m = manifest()
    for case in m['cases']:
        folder = run.cycle.folder_for(tmp_path, case); folder.mkdir(parents=True)
        run.resume.atomic_npz(folder/'base-response.npz', response=np.full((16, 5, 302), .001))
        run.resume.atomic_json(folder/'extension.json', dict(cycle_mean_time_s=list(np.linspace(.5, 31.5, 28))))
    for task in run.tasks_for(m):
        if task['begin'] == 0:
            save_batch(tmp_path, task)
    result = run.evaluate(tmp_path, m)
    assert result['covered_target_checks_passed'] and not result['full_target_qualified']
    assert result['completed_targets'] == ['A', 'B', 'C']
    for task in run.tasks_for(m):
        if task['begin'] != 0:
            save_batch(tmp_path, task)
    result = run.evaluate(tmp_path, m)
    assert result['full_target_complete'] and result['full_target_qualified']
    task = run.tasks_for(m)[-1]
    save_batch(tmp_path, task, .1)
    assert not run.evaluate(tmp_path, m)['full_target_qualified']


def test_verified_copy_refuses_changed_source_and_preserves_existing(tmp_path):
    src = tmp_path/'src'; dst = tmp_path/'dst'; src.write_bytes(b'original')
    inputs = {}
    run.copy_verified(src, dst, inputs)
    src.write_bytes(b'changed')
    with pytest.raises(ValueError):
        run.copy_verified(src, dst, inputs)
    assert dst.read_bytes() == b'original'


def test_prepare_reuses_audited_panel_and_all_original_responses_without_source_changes(tmp_path):
    source = tmp_path/'source'; ancestor = tmp_path/'ancestor'; root = tmp_path/'new'
    source.mkdir(); ancestor.mkdir(); root.mkdir()
    targets = ['T'+str(i) for i in range(135)]
    panel = targets[:3]
    model = source/'peptide/model.npz'; model.parent.mkdir()
    np.savez_compressed(model, neuron_ids=np.array(['RMDL', 'RMDR']+[str(i) for i in range(300)]))
    model_hash = run.file_sha256(model)
    cases = []
    for level in (0, 1):
        case = dict(arm='peptide', level=level, model_sha256=model_hash, tolerance=1e-5)
        cases.append(case)
        folder = run.cycle.folder_for(source, case); folder.mkdir()
        for name in ('base-starts.npz', 'extension.npz', 'extension-states.npz', 'extension-end.npz'):
            run.resume.atomic_npz(folder/name, voltage_mV=np.zeros((16, 302)))
        for name in ('base-phases.json', 'base-metrics.json', 'extension.json'):
            run.resume.atomic_json(folder/name, {})
        old = ancestor/'peptide'/f'full-{level}'
        run.resume.atomic_npz(old/'response.npz', response=np.full((16, 135, 302), level*.00001))
        for group in range(7):
            save_batch(source, dict(arm='peptide', level=level, group=group, begin=0, targets=panel, model_sha256=model_hash))
    run.resume.atomic_json(ancestor/'manifest.json', dict(full_targets=targets))
    run.resume.atomic_json(ancestor/'audit.json', dict(artifact_sha256={str(p): run.file_sha256(p) for p in ancestor.rglob('*.npz')}))
    run.resume.atomic_json(source/'manifest.json', dict(source_run=str(ancestor), panel=panel, cases=cases, source_sha256={}, input_sha256={}, worker_cpus=[[0], [1]], numeric_tolerance=dict(peptide=1e-5)))
    run.resume.atomic_json(source/'audit.json', dict(audit_complete=True, execution_complete=True, integrity_passed=True,
        stages=[dict(cycles=32, cases=[dict(arm='peptide', sampling=dict(passed=True), controls_and_dynamics_passed=True)], numerical=[dict(arm='peptide', passed=True)])],
        artifact_sha256={str(p): run.file_sha256(p) for p in source.rglob('*.npz')}))
    run.resume.atomic_json(source/'budget_receipt.json', dict(all_steps_finished=True))
    before = {str(p): run.file_sha256(p) for p in source.rglob('*') if p.is_file()}
    run.prepare(root, source)
    m = json.loads((root/'manifest.json').read_text())
    assert len(m['reused_tasks']) == 14 and len(run.tasks_for(m)) == 476
    assert len(list(root.glob('peptide/level-*/batches/*.npz'))) == 14
    for case in m['cases']:
        path = run.cycle.folder_for(root, case)/'base-response.npz'
        assert run.resume.old.prior.arrays(path)['response'].shape == (16, 135, 302)
    run.verify_inputs(m)
    assert all(run.file_sha256(p) == h for p, h in before.items())
    # Re-preparation after interruption is idempotent and never replaces sources.
    run.prepare(root, source)


def test_main_has_no_validation_timeout_and_completed_rerun_does_nothing(tmp_path, monkeypatch):
    root = tmp_path/'job'; source = tmp_path/'source'
    run.resume.atomic_json(root/'manifest.json', dict(input_sha256={}, source_sha256={}))
    monkeypatch.setattr(run, 'BASE', tmp_path)
    monkeypatch.setattr(run, 'jobs', lambda: [])
    monkeypatch.setattr(run, 'capture_run_record', lambda *a, **k: None)
    calls = []
    def work(path):
        calls.append(path)
        assert json.loads((path/'job.json').read_text())['full_validation_admitted']
        return 0
    monkeypatch.setattr(run, 'run_workers', work)
    monkeypatch.setattr(run, 'audit', lambda path: dict(audit_complete=True, integrity_passed=True, full_target_complete=True, full_target_qualified=False))
    monkeypatch.setattr(sys, 'argv', ['full', '--output-dir', str(root), '--source-run', str(source)])
    assert run.main() == 0
    receipt = json.loads((root/'budget_receipt.json').read_text())
    assert receipt['planning_minutes'] == 480
    assert receipt['validation_wallclock_cutoff'] is False and receipt['all_steps_finished']
    assert run.main() == 0 and len(calls) == 1
