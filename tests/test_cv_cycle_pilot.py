import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from scripts import cv_cycle_pilot_core as core
from scripts import run_cv_cycle_pilot as run


def origin():
    base = core.arrays(Path(__file__).resolve().parents[1]/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz')
    base['peptide_raw_exc'] = np.full(3, -9.)
    base['peptide_raw_inh'] = np.full(3, -9.)
    return base


def folds():
    pairs = [[name, 'R'] for name in ('A', 'B', 'C', 'D', 'E') for _ in range(5)]
    # eligible_targets counts distinct receivers, not repeated trials.
    pairs = [[name, 'R'+str(i)] for name in ('A', 'B', 'C', 'D', 'E') for i in range(5)]
    return [dict(stats=dict(train=dict(pairs=pairs), heldout=dict(pairs=[], mean_z=[]))) for _ in range(5)]


def test_panel_uses_training_eligibility_not_heldout_values_and_rng_is_durable():
    rows = folds(); panel, state = core.panel_for(rows, list('ABCDE'), return_state=True)
    changed = copy.deepcopy(rows)
    for fold in changed:
        fold['stats']['heldout'] = dict(pairs=[['Z', 'R']], mean_z=[1e99])
    assert core.panel_for(changed, list('ABCDE')) == panel
    assert len(panel) == len(set(panel)) == 4
    rng = np.random.default_rng(); rng.bit_generator.state = state
    assert rng.bit_generator.state == state


def test_parameter_groups_are_bounded_and_known_fields_stay_exact():
    base = origin(); coeff = [.01, -.02, .03, -.04, .05]
    actual, x = core.decode(base, coeff)
    audit = core.parameter_audit(base, actual, coeff)
    assert audit['passed'] and audit['changed_neural_coordinates'] > 0
    assert not np.array_equal(actual['peptide_raw_exc'], base['peptide_raw_exc'])
    for key in base:
        if key not in ('params_log_channel_scale', 'params_raw_chemical', 'params_raw_gap', 'peptide_raw_exc', 'peptide_raw_inh'):
            assert np.array_equal(base[key], actual[key])
    actual['fixed_capacitance_pF'] = np.ones(302)  # unexpected physical field also fails
    assert not core.parameter_audit(base, actual, coeff)['passed']
    with pytest.raises(ValueError):
        core.decode(base, [.11, 0, 0, 0, 0])


def test_failed_or_unsafe_training_proposals_never_replace_parent():
    current = dict(name='parent', score=1., safe=True)
    proposals = [dict(name='better-but-unsafe', score=.1, safe=False), dict(name='failed', score=None, safe=False)]
    assert core.choose(current, proposals) == current
    assert core.choose(current, proposals+[dict(name='good', score=.9, safe=True)])['name'] == 'good'


def test_sampling_temporal_drift_rejected_even_when_leaveout_small():
    y = np.ones((128, 4, 302))*.001
    y[64:] *= 1.2
    result = core.check_response(y, list(range(32)), .00005)
    assert result['leave_one_out_passed'] and not result['temporal_halves_passed'] and not result['passed']
    assert core.numerical(y, y+.001, .00001)['passed'] is False


def test_status_selection_and_report_do_not_create_or_start_work(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'BASE', tmp_path); monkeypatch.setattr(run, 'jobs', lambda: [])
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **k: pytest.fail('Unexpected job'))
    for args in (['--status'], ['--select-output'], ['--output-dir', str(tmp_path/'absent'), '--report']):
        monkeypatch.setattr(sys, 'argv', ['pilot']+args)
        assert run.main() == 0
    assert list(tmp_path.iterdir()) == []


def test_active_gpu_job_refused_before_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'jobs', lambda: [dict(pid=1)])
    monkeypatch.setattr(sys, 'argv', ['pilot', '--output-dir', str(tmp_path/'new')])
    with pytest.raises(SystemExit):
        run.main()
    assert not (tmp_path/'new').exists()


def test_post_audit_pruning_keeps_every_parameter_optimizer_and_log(tmp_path):
    model = tmp_path/'models/proposal.npz'; model.parent.mkdir(); model.write_bytes(b'weights')
    optimizer = tmp_path/'optimizer.npz'; optimizer.write_bytes(b'optimizer')
    root = tmp_path/'evaluations/candidate'; root.mkdir(parents=True)
    for name in ('starts.npz', 'response.npz', 'batch-00.npz'):
        (root/name).write_bytes(b'raw array')
    (root/'result.json').write_text('{}')
    hashes = {str(p): run.file_sha256(p) for p in tmp_path.rglob('*.npz')}
    audit = dict(audit_complete=True, integrity_passed=True, execution_complete=True, artifact_sha256=hashes)
    run.prune_temporary(tmp_path, audit)
    assert model.read_bytes() == b'weights' and optimizer.read_bytes() == b'optimizer'
    assert (root/'result.json').exists() and not list(root.glob('*.npz'))
    events = [json.loads(line) for line in (tmp_path/'retention.jsonl').read_text().splitlines()]
    assert sum(e['event'] == 'deleted' for e in events) == 3
    assert all(len(e['sha256']) == 64 for e in events)
    run.prune_temporary(tmp_path, audit)


def test_no_array_pruning_before_clean_complete_audit(tmp_path):
    folder = tmp_path/'evaluations/c'; folder.mkdir(parents=True)
    (folder/'response.npz').write_bytes(b'keep')
    run.prune_temporary(tmp_path, dict(audit_complete=True, integrity_passed=False, execution_complete=True))
    assert (folder/'response.npz').exists()


def test_main_validation_has_no_deadline_and_completed_launch_is_noop(tmp_path, monkeypatch):
    from scripts import total_job_budget
    root = tmp_path/'job'; root.mkdir()
    run.io.atomic_json(root/'manifest.json', {})
    monkeypatch.setattr(run, 'BASE', tmp_path); monkeypatch.setattr(run, 'jobs', lambda: [])
    monkeypatch.setattr(run, 'verify', lambda *a: {})
    monkeypatch.setattr(run, 'capture_run_record', lambda *a, **k: None)
    monkeypatch.setattr(run, 'prune_temporary', lambda *a: None)
    calls = []
    monkeypatch.setattr(total_job_budget, 'run_group', lambda command, seconds: (calls.append(seconds) or (0, False)))
    monkeypatch.setattr(run, 'audit', lambda *a: dict(audit_complete=True, execution_complete=True, integrity_passed=True, pilot_ready_for_longer_fit=False))
    monkeypatch.setattr(sys, 'argv', ['pilot', '--output-dir', str(root)])
    assert run.main() == 0 and calls == [None]
    assert run.main() == 0 and calls == [None]
    assert json.loads((root/'budget_receipt.json').read_text())['validation_wallclock_cutoff'] is False


def test_engine_reuses_completed_batches_and_own_state_checkpoint(tmp_path, monkeypatch):
    import jax
    from r_elegans.brain.peptide_state_replay import FullState
    from r_elegans.brain.conductance_network import ConductanceNetworkState
    from r_elegans.brain import peptide_checkpoint, midpoint_replay, cycle_ensemble
    ids = ['RMDL', 'RMDR']+[f'N{i}' for i in range(300)]
    calls = dict(anchor=0, extension=0, replay=0)
    def state(n):
        return FullState(ConductanceNetworkState(np.full((n, 302), -30.), np.zeros((n, 302, 35)), np.zeros((n, 302)), np.zeros((n, 302))), np.zeros((n, 302)))
    def anchors(*args):
        calls['anchor'] += 1
        return state(16), dict(indices=list(range(16)), reference_voltage_mV=[-30.]*4000)
    def extension(*args):
        calls['extension'] += 1
        return state(112), state(1), dict(finite=True, voltage_min_mV=-30., voltage_max_mV=-30., cycle_mean_time_s=list(np.linspace(.5, 31.5, 28)))
    def replay(*args):
        calls['replay'] += 1
        if calls['replay'] == 4:
            raise RuntimeError('simulated interruption')
        means = np.zeros((16, 6, 302)); means[:, :4] = .001
        return means, np.array(0.), (np.array(0.),)*6
    monkeypatch.setattr(peptide_checkpoint, 'load_model', lambda path: (None, None, None, None, None, None, None, dict(neuron_ids=np.asarray(ids))))
    monkeypatch.setattr(midpoint_replay, 'make_anchors', lambda *a, **k: anchors)
    monkeypatch.setattr(midpoint_replay, 'make_replay', lambda *a, **k: replay)
    monkeypatch.setattr(cycle_ensemble, 'make_extension', lambda *a, **k: extension)
    monkeypatch.setattr(core, 'batch_metrics', lambda *a: dict(passed=True))
    run.io.atomic_json(tmp_path/'manifest.json', dict(panel=ids[2:6], sampling_tolerances={'0':1e-5}))
    (tmp_path/'model.npz').write_bytes(b'parameters')
    task = dict(id='candidate', model='model.npz', model_sha256=run.file_sha256(tmp_path/'model.npz'), level=0)
    engine = core.Engine(tmp_path)
    with pytest.raises(RuntimeError):
        engine.evaluate(task)
    assert len(list((tmp_path/'evaluations/candidate').glob('batch-*.npz'))) == 3
    result = core.Engine(tmp_path).evaluate(task)
    assert calls == dict(anchor=1, extension=1, replay=9)
    assert result['safe'] and result['sampling']['passed']


def test_expired_optimization_still_finishes_fivefold_validation_and_audit(tmp_path, monkeypatch):
    from r_elegans.brain.peptide_checkpoint import bundle_model
    from scripts.experiment_records import file_sha256
    base = core.arrays(Path(__file__).resolve().parents[1]/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz')
    net = tmp_path/'network.npz'; np.savez_compressed(net, **base)
    saved = tmp_path/'start.npz'
    np.savez_compressed(saved, start_voltage_mV=np.full((4, 302), -30.), start_gates=np.zeros((4, 302, 35)),
                        start_calcium_uM=np.zeros((4, 302)), start_indicator=np.zeros((4, 302)), start_peptide=np.zeros((4, 302)))
    asset = core.arrays(Path(__file__).resolve().parents[1]/'r_elegans/assets/extrasynaptic_v1.npz')
    bundle_model(net, saved, tmp_path/'parent-model.npz', enabled=True, asset=asset, asset_sha256='asset', network_sha256=file_sha256(net))
    a = core.arrays(tmp_path/'parent-model.npz'); ids = a['neuron_ids'].tolist(); panel = ids[:4]
    y = np.broadcast_to(np.arange(4*302).reshape(1, 4, 302)*1e-8, (128, 4, 302)).copy()
    pairs = [[t, n] for t in panel for n in ids[4:10]]
    stats = dict(pairs=pairs, mean_z=list(np.linspace(-1, 1, len(pairs))))
    rows = [dict(stats=dict(train=stats, heldout=stats)) for _ in range(5)]
    run.io.atomic_json(tmp_path/'folds.json', rows)
    m = dict(panel=panel, numerical_tolerance=1e-5, sampling_tolerances={'0':1e-5, '1':1e-5})
    run.io.atomic_json(tmp_path/'manifest.json', m)
    run.io.atomic_json(tmp_path/'job.json', dict(optimization_minutes=180))
    scores = [core.score(y, panel, ids, fold | {'_origin': a}, [0.]*5) for fold in rows]
    state = dict(folds=[dict(coefficients=[0.]*5, selected='parent-0', probes=0, score=s) for s in scores], history=[], next_fold=0, deadline_unix=0.)
    run.io.atomic_json(tmp_path/'optimizer.json', state)
    for level in (0, 1):
        folder = tmp_path/'evaluations'/f'parent-{level}'
        run.io.atomic_npz(folder/'response.npz', response=y)
        task = dict(id=f'parent-{level}', model='parent-model.npz', model_sha256=file_sha256(tmp_path/'parent-model.npz'), level=level)
        run.io.atomic_json(folder/'result.json', dict(task=task, status='completed', safe=True, sampling=core.check_response(y, list(range(32)), 1e-5),
            reused_source=True, source_sha256={}, response_sha256=file_sha256(folder/'response.npz'), elapsed_seconds=0., batch_checks=[],
            controls_from_audited_source=True, cycle_times=list(range(32))))
    monkeypatch.setattr(run, 'verify', lambda root: m)
    class Process:
        def poll(self): return None
        def wait(self, **kwargs): return 0
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **k: Process())
    assert run.coordinate(tmp_path) is None
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['validation_complete'] and len(summary['folds']) == 5
    assert not summary['pilot_descent_demonstrated'] and summary['optimization_probes'] == 0
    assert all(file_sha256(tmp_path/f'fold-{i}/selected-model.npz') == file_sha256(tmp_path/'parent-model.npz') for i in range(5))
    audited = run.audit(tmp_path)
    assert audited['integrity_passed'] and audited['execution_complete'] and not audited['pilot_ready_for_longer_fit']
