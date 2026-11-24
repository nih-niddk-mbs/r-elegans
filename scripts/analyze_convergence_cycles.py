"""Analyze saved E107/E123 cycle windows; source read-only, no simulation or changed gates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from scripts.experiment_records import capture_run_record, file_sha256
from scripts.selected_atlas_audit_checks import clean, saved_arrays, write


def peak_series(voltage, dt):
    v = np.asarray(voltage, dtype=float)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Expected a finite voltage series and positive recording step')
    # E107 records state after each neural interval.
    t = (np.arange(len(v)) + 1) * dt
    ix = find_peaks(v, prominence=1., distance=max(1, round(.02 / dt)))[0]
    curvature = v[ix - 1] - 2 * v[ix] + v[ix + 1]
    shift = np.divide(.5 * (v[ix - 1] - v[ix + 1]), curvature,
                      out=np.zeros(len(ix)), where=curvature != 0)
    interpolated = t[ix] + np.clip(shift, -.5, .5) * dt
    return t, ix, interpolated


def window_stats(voltage, dt, start, end):
    v = np.asarray(voltage, dtype=float)
    t, ix, refined = peak_series(v, dt)
    # Boundary peaks are detected on the entire saved series, not truncated windows.
    use = (t[ix] >= start) & (t[ix] < end)
    ix, refined = ix[use], refined[use]
    result = dict(window_seconds=[start, end], peaks=len(ix),
                  peak_times_sampled_s=t[ix].tolist(), peak_times_interpolated_s=refined.tolist())
    if len(ix) < 3:
        return dict(result, available=False, reason='fewer_than_three_peaks')
    periods = np.diff(refined)
    sampled = np.diff(t[ix])
    slope, intercept = np.polyfit(np.arange(len(ix)), refined, 1)
    cycle_waves, amplitudes = [], []
    for first, last in zip(ix[:-1], ix[1:]):
        segment = v[first:last + 1]
        cycle_waves.append(np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, len(segment)), segment))
        amplitudes.append(np.ptp(segment))
    periods_last = periods[-5:]
    return clean(dict(result, available=True, periods_s=periods,
        cycle_mid_times_s=.5 * (refined[1:] + refined[:-1]),
        sampled_mean_period_s=float(sampled.mean()), sampled_median_period_s=float(np.median(sampled)),
        interpolated_mean_period_s=float(periods.mean()), interpolated_median_period_s=float(np.median(periods)),
        period_quantiles_5_50_95_s=np.quantile(periods, [.05, .5, .95]),
        period_cv=float(periods.std() / periods.mean()),
        whole_train_period_s=float(slope), whole_train_frequency_hz=float(1 / slope),
        whole_train_peak_time_residual_rmse_s=float(np.sqrt(np.mean((refined - (slope * np.arange(len(ix)) + intercept)) ** 2))),
        last_five_count=len(periods_last), last_five_sampled_mean_period_s=float(sampled[-5:].mean()),
        last_five_interpolated_mean_period_s=float(periods_last.mean()),
        mean_cycle_amplitude_mV=float(np.mean(amplitudes)), cycle_amplitudes_mV=amplitudes,
        mean_cycle_voltage_mV=np.mean(cycle_waves, axis=0)))


def descriptive_comparison(coarse, fine):
    if not coarse['available'] or not fine['available']:
        return dict(available=False, reason='insufficient_peaks')
    return dict(available=True,
        whole_train_period_relative_difference=abs(coarse['whole_train_period_s'] / fine['whole_train_period_s'] - 1),
        full_window_mean_amplitude_relative_difference=abs(coarse['mean_cycle_amplitude_mV'] / max(fine['mean_cycle_amplitude_mV'], 1e-12) - 1),
        full_window_cycle_rmse_mV=float(np.sqrt(np.mean((np.asarray(coarse['mean_cycle_voltage_mV']) - fine['mean_cycle_voltage_mV']) ** 2))),
        scope='Descriptive full-window comparison only; does not replace last-five-cycle gates')


def analyze(run, out):
    kind=json.loads((run/'manifest.json').read_text()).get('experiment')
    if kind in ('E123','E123-continuation'):
        return analyze_validation(run,out)
    evidence = {}
    def read(path, expected=None):
        path = Path(path)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('Input checksum mismatch: ' + str(path))
        evidence[str(path)] = digest
        return path
    manifest_path = read(run / 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    original = json.loads(read(run / 'audit.json').read_text())
    budget = json.loads(read(run / 'budget_receipt.json').read_text())
    if not original.get('audit_complete') or not original.get('integrity_passed') or not budget.get('all_steps_finished'):
        raise ValueError('Requires the completed E107 source audit with passing integrity')
    if set(manifest['models']) != {'selected', 'parent'}:
        raise ValueError('Expected both predetermined E107 models')
    for path, digest in manifest['evidence_sha256'].items():
        read(path, digest)
    models = {}
    for label, model in manifest['models'].items():
        read(model['path'], model['sha256'])
        summary = json.loads(read(run / label / 'summary.json').read_text())
        if summary['manifest_sha256'] != evidence[str(manifest_path)] or summary['checkpoint_sha256'] != model['sha256']:
            raise ValueError('Worker model/manifest identity mismatch')
        if len(summary['levels']) != 5 or len(original['models'][label]['levels']) != 5:
            raise ValueError('All five E107 levels required')
        levels = []
        for i, row in enumerate(summary['levels']):
            if row['level'] != i:
                raise ValueError('Unexpected level order')
            path = run / label / ('level-' + str(i) + '.npz')
            if Path(row['artifact']).resolve() != path.resolve():
                raise ValueError('Unexpected artifact path')
            read(path, row['sha256'])
            audited = original['models'][label]['levels'][i]
            if audited['level'] != i or audited['artifact_sha256'] != row['sha256']:
                raise ValueError('Artifact differs from E107 audited input')
            data = saved_arrays(path)
            voltage = data['rmd_voltage_mV']
            dt = float(data['sample_dt_s'])
            if voltage.shape != (20000, 2) or dt != .002 or not np.isfinite(voltage).all():
                raise ValueError('Unexpected trace shape, recording step or nonfinite voltage')
            if float(data['rhythm_step_ms']) != audited['rhythm_step_ms']:
                raise ValueError('Rhythm grid metadata differs from original audit')
            cells = {}
            for k, name in enumerate(('RMDL', 'RMDR')):
                cells[name] = dict(full_late=window_stats(voltage[:, k], dt, 20., 40.),
                    windows=[window_stats(voltage[:, k], dt, t, t + 5.) for t in (20., 25., 30., 35.)])
            levels.append(dict(level=i, rhythm_step_ms=audited['rhythm_step_ms'], cells=cells))
        comparisons = []
        for i in range(1, 5):
            comparisons.append(dict(levels=[i - 1, i], cells={
                name: descriptive_comparison(levels[i - 1]['cells'][name]['full_late'], levels[i]['cells'][name]['full_late'])
                for name in ('RMDL', 'RMDR')}))
        models[label] = dict(checkpoint_sha256=model['sha256'], levels=levels, descriptive_comparisons=comparisons,
                             original_comparisons=original['models'][label]['comparisons'])
    for path, digest in evidence.items():
        if file_sha256(path) != digest:
            raise ValueError('Source changed during analysis: ' + path)
    result = dict(source_run=str(run), analysis_complete=True, integrity_passed=True, evidence_sha256=evidence,
        original_execution_budget=budget, original_numerical_component_passed=original['numerical_component_passed'],
        models=models, acceptance_gates_changed=False, model_promoted=False,
        limits=['Descriptive saved RMD-voltage analysis only; no forward simulation, fitting or changed thresholds.',
                'Peak interpolation is an estimate within recorded2ms sampling, not reconstruction of sub-sample voltage.',
                'Only atlas mean differences and RMD traces were saved; no full neural state or stimulus-locked voltage trajectories.',
                'Phase drift is not a proven cause of atlas failures. No phase-conditioned cue test or independent solver here.',
                'No biological, body gait, sensory start/stop or swimming qualification.'])
    write(out / 'summary.json', result)
    return result


def analyze_validation(run,out):
    """Use all late cycles of the existing300s E123 traces; no new pass flag."""
    evidence={}
    def read(path,expected=None):
        path=Path(path);digest=file_sha256(path)
        if expected is not None and digest!=expected:raise ValueError('Input checksum mismatch: '+str(path))
        evidence[str(path)]=digest;return path
    manifest=json.loads(read(run/'manifest.json').read_text())
    audit=json.loads(read(run/'audit.json').read_text())
    budget=json.loads(read(run/'budget_receipt.json').read_text())
    if not (audit.get('audit_complete') and audit.get('execution_complete') and audit.get('integrity_passed') and budget.get('all_steps_finished')):
        raise ValueError('Completed E123 execution with passing integrity required; scientific gates may fail')
    if set(manifest['model_sha256'])!={'network','peptide'}:raise ValueError('Both predetermined model arms required')
    models={}
    for label,sha in manifest['model_sha256'].items():
        read(run/label/'model.npz',sha)
        if audit['models'][label]!=sha:raise ValueError('Original audited model differs')
        summary=json.loads(read(run/label/'summary.json').read_text())
        levels=[]
        if len(summary['long'])!=2:raise ValueError('Two complete long clocks required')
        for level,row in enumerate(summary['long']):
            dt=.001/2**level
            if row['level']!=level or row['dt']!=dt:raise ValueError('Unexpected long clock')
            path=run/label/f'long-{level}.npz'
            if str(path) not in audit['artifact_sha256']:raise ValueError('Original artifact hash unavailable')
            data=saved_arrays(read(path,audit['artifact_sha256'][str(path)]));v=data['rmd']
            if v.shape!=(round(300./dt),2) or not np.isfinite(v).all():raise ValueError('Unexpected300s voltage data')
            cells={name:dict(full_late=window_stats(v[:,i],dt,150.,300.),
                windows=[window_stats(v[:,i],dt,t,t+50.) for t in (150.,200.,250.)]) for i,name in enumerate(('RMDL','RMDR'))}
            levels.append(dict(level=level,rhythm_step_ms=.03125/2**level,sample_dt_s=dt,cells=cells))
        comparisons=[dict(levels=[0,1],cells={name:descriptive_comparison(levels[0]['cells'][name]['full_late'],levels[1]['cells'][name]['full_late']) for name in ('RMDL','RMDR')})]
        windows=[dict(window_seconds=[t,t+50.],cells={name:descriptive_comparison(levels[0]['cells'][name]['windows'][i],levels[1]['cells'][name]['windows'][i]) for name in ('RMDL','RMDR')}) for i,t in enumerate((150.,200.,250.))]
        models[label]=dict(checkpoint_sha256=sha,levels=levels,descriptive_comparisons=comparisons,
            window_comparisons=windows,original_comparisons=summary['long_cycle_checks'],original_checks_passed=summary['checks_passed'])
    for path,sha in evidence.items():
        if file_sha256(path)!=sha:raise ValueError('Source changed during analysis: '+path)
    result=dict(source_run=str(run),source_experiment=manifest['experiment'],analysis_complete=True,integrity_passed=True,
        evidence_sha256=evidence,models=models,original_execution_budget=budget,
        original_numerical_component_passed=audit['endpoint_checks_passed'],acceptance_gates_changed=False,model_promoted=False,
        limits=['Descriptive analysis of saved voltage only; outputs saved separately, source records unchanged.',
            '150s aggregate and50s windows do not replace original last-five-cycle gates or prove convergence.',
            'Peak interpolation estimates timing within saved samples; no independent solver or new trajectories.',
            'Different grid-specific own-state anchors and slow modulation can affect comparisons; causation unproven.',
            'A third long-trace grid, biological fidelity, full atlas/CV fitting and body/sensory start-stop are not established.'])
    write(out/'summary.json',result)
    return result


def plot_periods(result, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return dict(available=False, reason='matplotlib unavailable; JSON contains all cycle statistics')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for row, (label, model) in enumerate(result['models'].items()):
        for col, name in enumerate(('RMDL', 'RMDR')):
            ax = axes[row, col]
            for level in model['levels']:
                stats = level['cells'][name]['full_late']
                if stats['available']:
                    ax.plot(stats['cycle_mid_times_s'], stats['periods_s'], marker='.', linewidth=.6,
                            label=str(level['rhythm_step_ms']) + ' ms')
            ax.set_title(label + ' ' + name)
            ax.set_ylabel('Interpolated cycle period (s)')
            ax.set_xlabel('Time after recording begins (s)')
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle(result.get('source_experiment','E107')+' saved RMD cycles: descriptive timing; acceptance gates unchanged')
    fig.tight_layout()
    path = out / 'cycle-periods.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return dict(available=True, path=str(path), sha256=file_sha256(path))


def print_report(result, output):
    print('Source:', result['source_run'])
    for label, model in result['models'].items():
        for level in model['levels']:
            for name, cell in level['cells'].items():
                s = cell['full_late']
                print(label, 'step_ms', level['rhythm_step_ms'], name, {k: s.get(k) for k in (
                    'peaks', 'whole_train_period_s', 'interpolated_median_period_s',
                    'last_five_sampled_mean_period_s', 'last_five_interpolated_mean_period_s',
                    'period_cv', 'period_quantiles_5_50_95_s')},
                    'window_period_s', [w.get('whole_train_period_s') for w in cell['windows']])
        print(label, 'finest descriptive comparison:', model['descriptive_comparisons'][-1])
        if model.get('window_comparisons'):
            print(label,'50s descriptive comparisons:',model['window_comparisons'])
            print(label,'Original last-five-cycle gates:',model['original_comparisons'])
    print('Original numerical component pass:', result['original_numerical_component_passed'])
    print('Acceptance gates unchanged; no model promoted. No simulation or fitting performed.')
    print('Report:', output / 'summary.json')
    if result.get('figure', {}).get('available'):
        print('Figure:', result['figure']['path'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    run = args.run.resolve()
    out = (args.output_dir or run.with_name(run.name + '-cycles-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    repo = Path(__file__).resolve().parents[1]
    if out == run or run in out.parents or out in run.parents or out == repo or repo in out.parents or out in repo.parents:
        p.error('Separate external output required')
    out.mkdir(parents=True, exist_ok=False)
    capture_run_record(out, repo, args, phase='saved-convergence-cycle-analysis')
    try:
        result = analyze(run, out)
        result['figure'] = plot_periods(result, out)
        write(out / 'summary.json', result)
        print_report(result, out)
    except Exception as exc:
        write(out / 'failure.json', dict(analysis_complete=False, error=repr(exc), model_promoted=False))
        raise


if __name__ == '__main__':
    main()
