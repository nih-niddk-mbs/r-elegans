import json
import sys
import numpy as np
import pytest
from scripts import cv_wide_cycle_core as core
from scripts import run_cv_wide_cycle_fit as run


def test_wider_panel_training_only_and_no_heldout_label_selection():
    names = [f'T{i:02d}' for i in range(24)]
    pairs = [[name, f'R{i}'] for name in names for i in range(5)]
    folds = [dict(stats=dict(train=dict(pairs=pairs), heldout=dict(pairs=[], mean_z=[]))) for _ in range(5)]
    panel, state = core.panel_for(folds, names, return_state=True)
    assert len(panel) == 16 and len(set(panel)) == 16
    for fold in folds:
        fold['stats']['heldout'] = dict(pairs=[['bad', 'data']], mean_z=[1e99])
    assert core.panel_for(folds, names) == panel
    rng = np.random.default_rng(); rng.bit_generator.state = state
    assert rng.bit_generator.state == state
    assert len(core.panel_for(folds, names[:10])) == 10
    with pytest.raises(ValueError): core.panel_for(folds, names[:4])


def test_fast_but_numerically_failed_batch_shape_is_never_selected():
    cases = [dict(capacity=4, passed=True, warm_seconds=40.), dict(capacity=8, passed=True, warm_seconds=22.),
             dict(capacity=16, passed=False, warm_seconds=1.)]
    assert core.choose_capacity(cases) == 8
    cases[2]['passed'] = True
    assert core.choose_capacity(cases) == 16
    cases[2]['warm_seconds'] = float('nan')
    assert core.choose_capacity(cases) == 8


def test_status_report_selection_are_readonly(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'BASE', tmp_path); monkeypatch.setattr(run, 'jobs', lambda: [])
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **k: pytest.fail('Read-only launch'))
    for args in (['--status'], ['--select-output'], ['--output-dir', str(tmp_path/'none'), '--report']):
        monkeypatch.setattr(sys, 'argv', ['wide']+args); assert run.main() == 0
    assert not list(tmp_path.iterdir())


def test_active_work_refused_before_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'jobs', lambda: [dict(pid=123)])
    monkeypatch.setattr(sys, 'argv', ['wide', '--output-dir', str(tmp_path/'new')])
    with pytest.raises(SystemExit): run.main()
    assert not (tmp_path/'new').exists()


def setup_engine(tmp_path, monkeypatch, capacity=8):
    from r_elegans.brain.peptide_state_replay import FullState
    from r_elegans.brain.conductance_network import ConductanceNetworkState
    from r_elegans.brain import peptide_checkpoint, midpoint_replay, cycle_ensemble
    ids = ['RMDL', 'RMDR']+[f'N{i}' for i in range(300)]
    counts = dict(anchor=0, extension=0, replay=0)
    def state(n):
        return FullState(ConductanceNetworkState(np.full((n, 302), -30.), np.zeros((n, 302, 35)), np.zeros((n, 302)), np.zeros((n, 302))), np.zeros((n, 302)))
    def anchor(*args):
        counts['anchor'] += 1
        v = -30+10*np.sin(2*np.pi*3*np.arange(4000)*.001)
        ix, meta = midpoint_replay.cycle_phases(v, .001)
        values = state(16); values.neural.voltage_mV[:, 0] = v[ix]
        return values, dict(meta, reference_voltage_mV=v.tolist())
    def extension(*args):
        counts['extension'] += 1
        v = -30+10*np.sin(2*np.pi*3*np.arange(32000)*.001)
        ix, meta = cycle_ensemble.extension_phases(v, .001)
        values = state(112); values.neural.voltage_mV[:, 0] = v[ix]
        return values, state(1), dict(meta, finite=True, voltage_min_mV=-40., voltage_max_mV=-20.)
    def replay(*args):
        counts['replay'] += 1
        stimuli = np.asarray(args[-1]); slots = len(stimuli)
        means = np.broadcast_to(stimuli[None]*.001, (16, slots, 302)).copy()
        return means, np.array(0.), (np.array(0.),)*6
    monkeypatch.setattr(peptide_checkpoint, 'load_model', lambda path: (None, None, None, None, None, None, None, dict(neuron_ids=np.asarray(ids))))
    monkeypatch.setattr(midpoint_replay, 'make_anchors', lambda *a, **k: anchor)
    monkeypatch.setattr(midpoint_replay, 'make_replay', lambda *a, **k: replay)
    monkeypatch.setattr(cycle_ensemble, 'make_extension', lambda *a, **k: extension)
    monkeypatch.setattr(core, 'batch_metrics', lambda *a: dict(passed=True, finite=True, voltage_min_mV=-40., voltage_max_mV=-20., duplicate_mean_error=0., duplicate_voltage_error_mV=0., failed_rhythms=[]))
    run.io.atomic_json(tmp_path/'manifest.json', dict(panel=ids[2:18], sampling_tolerances={'0':1e-5}, parent_sha256='parent'))
    run.io.atomic_json(tmp_path/'batch-policy.json', {'0':dict(capacity=capacity)})
    np.savez_compressed(tmp_path/'model.npz', neuron_ids=np.asarray(ids))
    task = dict(id='candidate', model='model.npz', model_sha256=run.file_sha256(tmp_path/'model.npz'), level=0)
    return counts, task


def test_wider_engine_covers_every_target_and_resumes_all_saved_tiles(tmp_path, monkeypatch):
    counts, task = setup_engine(tmp_path, monkeypatch, 8)
    result = core.Engine(tmp_path).evaluate(task)
    assert result['safe'] and result['batch_capacity'] == 8
    assert counts == dict(anchor=1, extension=1, replay=16)
    y = core.arrays(tmp_path/'evaluations/candidate/response.npz')['response']
    assert y.shape == (128, 16, 302)
    for target in range(16): assert np.all(y[:, target, target+2] == .00003)
    core.Engine(tmp_path).evaluate(task)
    assert counts == dict(anchor=1, extension=1, replay=16)


def test_batch_benchmark_compares_same_full_panel_and_reuses_its_checkpoints(tmp_path, monkeypatch):
    counts, task = setup_engine(tmp_path, monkeypatch)
    task.update(id='benchmark-0', kind='benchmark')
    result = core.Engine(tmp_path).evaluate(task)
    assert result['safe'] and len(result['cases']) == 3
    assert all(c['passed'] and c['panel_difference'] == 0 and c['repeat_difference'] == 0 for c in result['cases'])
    assert counts == dict(anchor=1, extension=1, replay=14)
    again = core.Engine(tmp_path).evaluate(task)
    assert again['chosen_capacity'] == result['chosen_capacity']
    assert counts == dict(anchor=1, extension=1, replay=14)


def test_missing_parent_baseline_reuses_verified_benchmark_states_and_first_group(tmp_path, monkeypatch):
    counts, task = setup_engine(tmp_path, monkeypatch)
    m = json.loads((tmp_path/'manifest.json').read_text()); m['parent_sha256'] = task['model_sha256']
    run.io.atomic_json(tmp_path/'manifest.json', m)
    bench = core.Engine(tmp_path).evaluate(dict(task, id='benchmark-0', kind='benchmark'))
    capacity = bench['chosen_capacity']
    run.io.atomic_json(tmp_path/'batch-policy.json', {'0':dict(capacity=capacity)})
    row = core.Engine(tmp_path).evaluate(dict(task, id='parent-0'))
    assert row['safe']
    assert counts['anchor'] == counts['extension'] == 1
    assert counts['replay'] == 14+7*((16+capacity-1)//capacity)


def test_benchmark_audit_recomputes_equivalence_and_rejects_altered_saved_tile(tmp_path, monkeypatch):
    _, task = setup_engine(tmp_path, monkeypatch)
    m = json.loads((tmp_path/'manifest.json').read_text()); m['parent_sha256'] = task['model_sha256']
    run.io.atomic_json(tmp_path/'manifest.json', m)
    row = core.Engine(tmp_path).evaluate(dict(task, id='benchmark-0', kind='benchmark'))
    run.io.atomic_json(tmp_path/'batch-policy.json', {'0':dict(capacity=row['chosen_capacity'], source='benchmark-0')})
    assert run.verify_benchmark(tmp_path, row, m) == []
    path = tmp_path/'evaluations/benchmark-0/batch-benchmark-04-warm-000.npz'
    data = core.arrays(path); data['means'][0, 0, 0] += .001
    run.io.atomic_npz(path, **data)
    assert run.verify_benchmark(tmp_path, row, m)


def test_wide_coordinator_finishes_validation_after_deadline_and_reports_missing_heldout(tmp_path, monkeypatch):
    from r_elegans.brain.peptide_checkpoint import bundle_model
    base = core.arrays(run.REPO/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz')
    np.savez_compressed(tmp_path/'network.npz', **base)
    np.savez_compressed(tmp_path/'start.npz', start_voltage_mV=np.full((4, 302), -30.), start_gates=np.zeros((4, 302, 35)),
                        start_calcium_uM=np.zeros((4, 302)), start_indicator=np.zeros((4, 302)), start_peptide=np.zeros((4, 302)))
    asset = core.arrays(run.REPO/'r_elegans/assets/extrasynaptic_v1.npz')
    bundle_model(tmp_path/'network.npz', tmp_path/'start.npz', tmp_path/'parent-model.npz', enabled=True, asset=asset,
                 asset_sha256='asset', network_sha256=run.file_sha256(tmp_path/'network.npz'))
    parent_hash = run.file_sha256(tmp_path/'parent-model.npz'); monkeypatch.setattr(core, 'PARENT_SHA', parent_hash)
    origin = core.arrays(tmp_path/'parent-model.npz'); ids = origin['neuron_ids'].tolist(); panel = ids[:16]
    y = np.broadcast_to(np.arange(16*302).reshape(1, 16, 302)*1e-8, (128, 16, 302)).copy()
    pairs = [[t, n] for t in panel for n in ids[16:22]]
    statistics = dict(pairs=pairs, mean_z=list(np.linspace(-1, 1, len(pairs))))
    folds = [dict(stats=dict(train=statistics, heldout=statistics if i else dict(pairs=[], mean_z=[]))) for i in range(5)]
    run.io.atomic_json(tmp_path/'folds.json', folds)
    m = dict(panel=panel, parent_sha256=parent_hash, numerical_tolerance=1e-5, sampling_tolerances={'0':1e-5, '1':1e-5})
    run.io.atomic_json(tmp_path/'manifest.json', m); run.io.atomic_json(tmp_path/'job.json', dict(optimization_minutes=480))
    state = dict(folds=[dict(coefficients=[0.]*5, selected='parent-0', probes=0,
                           score=core.score(y, panel, ids, f | {'_origin':origin}, [0.]*5)) for f in folds],
                 history=[], next_fold=0, deadline_unix=0.)
    run.io.atomic_json(tmp_path/'optimizer.json', state)
    for level in (0, 1):
        folder = tmp_path/'evaluations'/f'parent-{level}'
        run.io.atomic_npz(folder/'response.npz', response=y)
        task = dict(id=f'parent-{level}', model='parent-model.npz', model_sha256=parent_hash, level=level)
        run.io.atomic_json(folder/'result.json', dict(task=task, status='completed', safe=True,
             sampling=core.check_response(y, list(range(32)), 1e-5), cycle_times=list(range(32)),
             reused_source=True, source_sha256={}, response_sha256=run.file_sha256(folder/'response.npz'),
             elapsed_seconds=0., batch_checks=[], controls_from_audited_source=True))
        folder = tmp_path/'evaluations'/f'benchmark-{level}'
        run.io.atomic_npz(folder/'response.npz', response=y[:16])
        task = dict(id=f'benchmark-{level}', kind='benchmark', model='parent-model.npz', model_sha256=parent_hash, level=level)
        run.io.atomic_json(folder/'result.json', dict(task=task, status='completed', safe=True, chosen_capacity=16,
             cases=[dict(capacity=16, passed=True, warm_seconds=1.)], response_sha256=run.file_sha256(folder/'response.npz')))
    monkeypatch.setattr(run, 'verify', lambda root: m)
    class Process:
        def poll(self): return None
        def wait(self): return 0
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **k: Process())
    monkeypatch.setattr(run, 'verify_benchmark', lambda *a: [])
    run.coordinate(tmp_path)
    result = run.audit(tmp_path)
    assert result['integrity_passed'] and result['execution_complete']
    assert not result['heldout_coverage_complete'] and not result['pilot_ready_for_longer_fit']
    assert result['batch_policy'] == {str(i):dict(capacity=16, source=f'benchmark-{i}') for i in (0, 1)}
    assert json.loads((tmp_path/'summary.json').read_text())['folds'][0]['heldout']['unavailable'] == 'No measured downstream labels'


def test_heldout_extension_uses_all_recorded_stimulus_identities_after_selection():
    fold = dict(stats=dict(heldout=dict(pairs=[['A', 'R'], ['X', 'X'], ['Z', 'R']], mean_z=[1, 2, 3])),
                animal_statistics={'worm':dict(pairs=[['Q', 'R']], mean_z=[1e99])})
    assert run.heldout_targets(fold, ['A'], ['A', 'X', 'Q', 'Z']) == ['Q', 'Z']
    fold['stats']['heldout']['mean_z'] = [-1e99]*3
    assert run.heldout_targets(fold, ['A'], ['A', 'X', 'Q', 'Z']) == ['Q', 'Z']


def test_extra_heldout_stimuli_reuse_same_models_states_without_repeating_warmup(tmp_path, monkeypatch):
    counts, task = setup_engine(tmp_path, monkeypatch, 8)
    core.Engine(tmp_path).evaluate(task)
    extra = ['N16', 'N17', 'N18', 'N19', 'N20', 'N21']
    row = core.Engine(tmp_path).evaluate(dict(task, id='heldout', targets=extra, starts_from='candidate'))
    assert row['safe'] and counts == dict(anchor=1, extension=1, replay=24)
    y = core.arrays(tmp_path/'evaluations/heldout/response.npz')['response']
    assert y.shape == (128, 6, 302)
    for i in range(6): assert np.allclose(y[:, i, i+18], .00003)


def test_expanded_scoring_merges_target_order_and_failed_gate_never_changes_frozen_choice(tmp_path):
    m = dict(panel=['A', 'B'])
    decision = dict(extra_targets=['Y'], accepted=True, heldout_task=dict(id='held'))
    parent_train = np.ones((128, 2, 302)); candidate_train = parent_train*2
    parent_extra = np.stack([np.full((128, 302), 10.), np.full((128, 302), 20.)], axis=1)
    folder = tmp_path/'evaluations/held'
    run.io.atomic_npz(folder/'response.npz', response=np.full((128, 1, 302), 30.))
    row = dict(status='completed', safe=True, response_sha256=run.file_sha256(folder/'response.npz'))
    run.io.atomic_json(folder/'result.json', row)
    names, prediction, parent, error = run.expanded_prediction(tmp_path, m, decision, parent_train, parent_extra, ['X', 'Y'], candidate_train)
    assert names == ['A', 'B', 'Y'] and error is None
    assert np.all(prediction[:, :2] == 2) and np.all(prediction[:, 2] == 30)
    assert np.all(parent[:, :2] == 1) and np.all(parent[:, 2] == 20)
    row['safe'] = False; run.io.atomic_json(folder/'result.json', row)
    _, prediction, _, error = run.expanded_prediction(tmp_path, m, decision, parent_train, parent_extra, ['X', 'Y'], candidate_train)
    assert prediction is None and error and decision['accepted'] is True
