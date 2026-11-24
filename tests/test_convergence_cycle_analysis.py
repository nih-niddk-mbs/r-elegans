import numpy as np
import pytest
import json

from scripts.analyze_convergence_cycles import window_stats, descriptive_comparison


def test_interpolated_full_train_recovers_incommensurate_period():
    dt, period = .002, .23317
    t = (np.arange(20000) + 1) * dt
    v = -20 + 10 * np.sin(2 * np.pi * t / period + .3)
    result = window_stats(v, dt, 20, 40)
    assert result['available']
    assert abs(result['whole_train_period_s'] - period) < 1e-7
    assert abs(result['last_five_interpolated_mean_period_s'] - period) < 1e-6
    assert result['period_cv'] < 1e-4
    assert len(result['mean_cycle_voltage_mV']) == 101
    assert all(20 <= x < 40 for x in result['peak_times_sampled_s'])


def test_window_statistics_reveal_changing_period():
    dt = .002
    t = (np.arange(20000) + 1) * dt
    v = 10 * np.sin(2 * np.pi * (4 * t + .01 * t ** 2))
    early = window_stats(v, dt, 20, 25)
    late = window_stats(v, dt, 35, 40)
    whole = window_stats(v, dt, 20, 40)
    assert early['whole_train_period_s'] > late['whole_train_period_s']
    assert whole['period_cv'] > .01
    assert abs(whole['whole_train_period_s'] - whole['last_five_interpolated_mean_period_s']) > .001
    comparison = descriptive_comparison(early, late)
    assert 'passed' not in comparison  # analysis must not invent replacement acceptance gates


def test_plateau_is_not_a_cycle_and_bad_inputs_are_rejected():
    result = window_stats(np.full(20000, -30.), .002, 20, 40)
    assert not result['available']
    assert result['peaks'] == 0
    assert not descriptive_comparison(result, result)['available']
    with pytest.raises(ValueError):
        window_stats(np.full(20, np.nan), .002, 0, 1)
    with pytest.raises(ValueError):
        window_stats(np.zeros(20), 0, 0, 1)


def test_e123_saved_trace_analysis_preserves_failed_gates_and_source(tmp_path):
    from scripts.analyze_convergence_cycles import analyze
    from scripts.experiment_records import file_sha256
    root=tmp_path/'source';root.mkdir();models={};hashes={}
    for label in ('network','peptide'):
        folder=root/label;folder.mkdir()
        np.savez(folder/'model.npz',identity=np.array(label));models[label]=file_sha256(folder/'model.npz')
        for level,dt in enumerate((.001,.0005)):
            t=(np.arange(round(300./dt))+1)*dt
            voltage=np.column_stack([-25.+10*np.sin(2*np.pi*t/.23317),-25.+9*np.sin(2*np.pi*t/.21913+.4)])
            path=folder/f'long-{level}.npz';np.savez_compressed(path,rmd=voltage);hashes[str(path)]=file_sha256(path)
        (folder/'summary.json').write_text(json.dumps(dict(long=[dict(level=0,dt=.001),dict(level=1,dt=.0005)],
            long_cycle_checks=[dict(passed=False),dict(passed=True)],checks_passed=False)))
    (root/'manifest.json').write_text(json.dumps(dict(experiment='E123-continuation',model_sha256=models)))
    (root/'audit.json').write_text(json.dumps(dict(audit_complete=True,execution_complete=True,integrity_passed=True,
        models=models,artifact_sha256=hashes,endpoint_checks_passed=False)))
    (root/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True,validation_passed=False)))
    before={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()};out=tmp_path/'analysis';out.mkdir()
    result=analyze(root,out)
    assert result['analysis_complete'] and result['integrity_passed']
    assert not result['acceptance_gates_changed'] and not result['model_promoted']
    assert result['original_numerical_component_passed'] is False
    assert result['models']['network']['original_comparisons'][0]['passed'] is False
    assert 'passed' not in result['models']['network']['descriptive_comparisons'][0]['cells']['RMDL']
    assert len(result['models']['peptide']['levels'][1]['cells']['RMDR']['windows'])==3
    assert before=={str(p):file_sha256(p) for p in root.rglob('*') if p.is_file()}
    with (root/'network'/'long-0.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='checksum mismatch'):analyze(root,out)
