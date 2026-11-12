import json
import subprocess
import sys

import numpy as np
import pytest

from r_elegans.signal_chain_audit import channel_statistics, summarize_trace
from scripts.analyze_alternation_signals import paired_difference, source_files


def test_saved_file_diagnostic_imports_no_jax():
    subprocess.run([sys.executable, '-c',
                    'import sys; import scripts.analyze_alternation_signals; assert "jax" not in sys.modules'],
                   check=True, capture_output=True, text=True)


def test_curved_monotonic_drift_is_not_raw_reversal():
    t = np.linspace(0, 12, 1201)
    drift = .02 * np.exp(-t / 8.)
    stats = channel_statistics(t, drift[:, None], 1e-5)
    assert stats['minimum_window_detrended_p2p'][0] > 1e-5
    assert stats['peaks'] == stats['troughs'] == [0]
    assert stats['reversing_indices'] == stats['repeating_indices'] == []
    assert stats['positive_variation'] == [0.]
    assert stats['negative_variation'][0] > 0
    assert stats['weaker_direction_variation_fraction'] == [0.]


def test_neural_rhythm_does_not_imply_body_rhythm():
    t = np.linspace(0, 12, 1201)
    wave = np.sin(2 * np.pi * t)
    trace = dict(dense_time_s=t, dense_voltage_mV=np.column_stack([wave, wave * 0, wave * 0]),
                 dense_calcium_uM=np.column_stack([wave * .01, wave * 0, wave * 0]),
                 dense_muscles=np.full((len(t), 95), .5),
                 dense_command=np.zeros((len(t), 11)), dense_filtered_drive=np.zeros((len(t), 11)),
                 dense_joints=np.tile((.02 * np.exp(-t / 8))[:, None], (1, 11)))
    stats = summarize_trace(trace, ['DB1', 'AWAL', 'RMDL'])
    assert stats['dense_voltage_mV']['repeating_neurons'] == ['DB1']
    assert stats['dense_voltage_mV']['named_groups']['numbered_body_motor_names']['repeating_neurons'] == ['DB1']
    assert stats['dense_muscles']['repeating_indices'] == []
    assert stats['dense_joints']['repeating_indices'] == []
    assert stats['dense_muscles']['activation_near_one_fraction'] == 0
    trace['dense_voltage_mV'] = trace['dense_voltage_mV'][:, :2]
    with pytest.raises(ValueError, match='channel count'):
        summarize_trace(trace, ['DB1', 'AWAL', 'RMDL'])


def test_invalid_times_and_paired_grids_rejected():
    t = np.linspace(0, 12, 1201)
    x = np.zeros((len(t), 1))
    bad = t.copy(); bad[500] = np.nan
    with pytest.raises(ValueError, match='Finite ordered'):
        channel_statistics(bad, x, 1e-5)
    bad = t.copy(); bad[500] = bad[499]
    with pytest.raises(ValueError, match='Finite ordered'):
        channel_statistics(bad, x, 1e-5)
    with pytest.raises(ValueError, match='grids differ'):
        paired_difference(dict(dense_time_s=t), dict(dense_time_s=t + .001))


def test_saved_case_inventory_requires_complete_source(tmp_path):
    metadata = dict(initial_iteration=0, selected_iteration=3, validation=[dict(case='active')])
    with pytest.raises(ValueError, match='Required trace missing'):
        source_files(tmp_path, metadata)
    for directory, labels in [('selected/iteration-000000', ('active', 'sham', 'dose-low')),
                              ('selected/iteration-000003', ('active', 'sham', 'dose-low')),
                              ('active', ('active', 'sham'))]:
        folder = tmp_path / directory; folder.mkdir(parents=True)
        for label in labels:
            (folder / (label + '-trajectory.npz')).write_bytes(b'fixture')
    assert len(source_files(tmp_path, metadata)) == 8
    metadata['validation'] = [dict(case='../escape')]
    with pytest.raises(ValueError, match='Unsafe validation'):
        source_files(tmp_path, metadata)


def test_wrong_checkpoint_aborts_before_output(tmp_path, monkeypatch):
    from scripts.analyze_alternation_signals import main
    run = tmp_path / 'source'; run.mkdir()
    (run / 'network_best.npz').write_bytes(b'wrong checkpoint')
    (run / 'summary.json').write_text(json.dumps(dict(status='completed', checkpoint_sha256='0' * 64)))
    (run / 'configuration.json').write_text(json.dumps(dict(stage='alternation', arguments=dict(
        precision='float64', training_cues='nominal-low-dose'))))
    out = tmp_path / 'audit'
    monkeypatch.setattr('sys.argv', ['analyze', '--run', str(run), '--output-dir', str(out),
                                   '--expected-sha256', '0' * 64])
    with pytest.raises(ValueError, match='Pinned checkpoint hash mismatch'):
        main()
    assert not out.exists()
