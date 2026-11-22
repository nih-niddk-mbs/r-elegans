"""Independent saved-data E113 audit, including incomplete/failed executions."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scripts.experiment_records import file_sha256
from scripts.selected_atlas_audit_checks import saved_arrays
from scripts.stable_atlas_worker import atomic_json as write
from scripts.stable_atlas_policy import Coordinates, loss, prior, preserved


def audit(root):
    from scripts.run_stable_atlas_fit import analyze_results
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    experiment = 'E113'
    problems = [];unavailable = [];hashes = {};model_checks = [];task_checks = [];parameter_counts = {}
    def read(path, expected=None):
        path = Path(path);sha = file_sha256(path);hashes[str(path)] = sha
        if expected is not None and sha != expected:raise ValueError('Hash mismatch: '+str(path))
        return path
    def document(path):return json.loads(read(path).read_text())
    manifest_path = root/'manifest.json';summary_path = root/'summary.json'
    try:
        m = document(manifest_path);experiment = m.get('experiment', 'E113');statistics = document(m['statistics']);origin = saved_arrays(read(m['parent'], m['parent_sha256']))
        codec = Coordinates(origin);ids = origin['neuron_ids'].tolist()
        read(m['anchors'], m['anchors_sha256'])
        own = m.get('own_state', False)
        for path, sha in m['evidence_sha256'].items():read(path, sha)
        for folder in sorted((root/'checkpoints').glob('*')):
            identity = document(folder/'identity.json');path = read(folder/'network.npz', identity['sha256'])
            values = saved_arrays(read(folder/'coordinates.npz', identity['coordinates_sha256']))['offsets']
            check = codec.audit(saved_arrays(path), values);model_checks.append(dict(model=str(path), **check))
            if not check['passed']:problems.append('Parameter policy failed: '+str(path))
        specs = {};results = {}
        for task_path in sorted((root/'tasks').glob('*/task.json')):
            spec = document(task_path);specs[spec['name']] = spec
            read(spec['model'], spec['model_sha256'])
            if spec['kind'] == 'replay':read(spec['anchors'], spec['anchors_sha256'])
            path = Path(spec['result'])
            if not path.exists():
                unavailable.append('Interrupted task has no result: '+spec['name']);continue
            r = document(path);results[spec['name']] = r
            if r['task'] != spec:raise ValueError('Task/result identity differs')
            if r['status'] != 'completed':
                problems.append('Task failed: '+spec['name']);continue
            if spec['kind'] in ('autonomy', 'anchors'):
                anchor = saved_arrays(read(r['anchors'], r['anchors_sha256']))
                if own and str(anchor.get('model_sha256', 'missing')) != spec['model_sha256']:
                    raise ValueError('State generation model mismatch')
                if spec['kind'] == 'anchors':
                    task_checks.append(dict(name=spec['name'], own_state_model_verified=True));continue
                metrics = []
                for step in (.03125, .015625):
                    a = saved_arrays(read(Path(spec['output'])/('rhythm-'+str(step)+'.npz')))
                    metrics.append([rhythm_metrics(a['rmd_mV'][:, j], .002) for j in range(2)])
                if metrics != r['rhythms']:raise ValueError('Autonomy metrics differ')
                task_checks.append(dict(name=spec['name'], rhythms=metrics));continue
            if own and not spec['name'].startswith('source-check-'):
                anchor = saved_arrays(spec['anchors'])
                if str(anchor.get('model_sha256', 'missing')) != spec['model_sha256']:
                    raise ValueError('Replay did not use own model states')
            a = saved_arrays(read(r['artifact'], r['sha256']));targets = spec['targets'];n = len(targets)
            count = 2*((n+7)//8)+(-n)%8
            labels = [f'batch-{begin//8}-condition-{k}' for begin in range(0, n, 8) for k in range(min(8, n-begin), 10)]
            if a['response'].shape != (4,n,len(ids)) or a['control_means'].shape != (count,4,len(ids)) or a['control_labels'].tolist() != labels:
                raise ValueError('Missing target/sham coverage')
            if a['targets'].tolist() != targets or a['neuron_ids'].tolist() != ids or str(a['anchors_sha256']) != spec['anchors_sha256'] or float(a['neural_step_ms']) != spec['step_ms']:
                raise ValueError('Replay identity metadata differs')
            for key in ('response','control_means','reference_sham_rmd_mV'):
                if a[key].dtype != np.float64 or not np.isfinite(a[key]).all():raise ValueError('Nonfinite/nonfloat64 signal')
            mean_error = float(abs(a['control_means']-a['control_means'][0]).max())
            trace_error = float(a['maximum_control_voltage_delta_mV'])
            rhythm = [[rhythm_metrics(a['reference_sham_rmd_mV'][p,:,j], .001) for j in range(2)] for p in range(4)]
            controls = mean_error <= 1e-12 and trace_error <= 1e-12
            if rhythm != r['rhythms'] or controls != r['controls_passed'] or mean_error != r['duplicate_mean_error'] or trace_error != r['duplicate_trace_error']:
                raise ValueError('Independent replay/control metrics differ')
            task_checks.append(dict(name=spec['name'], controls_passed=controls, rhythms=rhythm,
                maximum_response=float(abs(a['response']).max()), control_count=count))
        # Independently recompute every completed optimization pair, same-minibatch decision and validation selection.
        def panel(prefix, targets, split, levels=(1,2), final=False):
            rows = [r for name,r in results.items() if name.startswith(prefix+'-') and r['status'] == 'completed']
            metrics, _ = analyze_results(rows, m, statistics, ids, targets, list(levels), split, final=final)
            return metrics
        for decision_path in sorted((root/'rounds').glob('*/decision.json')):
            d = document(decision_path);i = d['iteration'];batch = d['batch'];a = panel(f'r{i}-plus', batch, 'train');b = panel(f'r{i}-minus', batch, 'train')
            if a != d['plus_metrics'] or b != d['minus_metrics']:raise ValueError('Perturbation score differs')
            expected = False
            if 'proposal' in d:
                c = panel(f'r{i}-current', batch, 'train');v = panel(f'r{i}-proposal', batch, 'train')
                if c != d['current_metrics'] or v != d['proposal_metrics']:raise ValueError('Acceptance score differs')
                cx = saved_arrays(Path(d['current_before']).parent/'coordinates.npz')['offsets']
                px = saved_arrays(Path(d['proposal']).parent/'coordinates.npz')['offsets']
                expected = bool(a['passed'] and b['passed'] and c['passed'] and v['passed'] and v['data_loss']+prior(px) < c['data_loss']+prior(cx))
            if expected != d['accepted']:raise ValueError('Wrong update acceptance')
        baseline = saved_arrays(read(m['baselines'][-1]))['response']
        expected_score = loss(baseline, m['targets'], ids, statistics, 'val')
        if own:
            baseline_record = document(root/'own-parent-baseline.json')
            bm = panel('own-parent-baseline', baseline_record['metrics']['targets'], 'val')
            if bm != baseline_record['metrics'] or not bm['passed']:raise ValueError('Own parent baseline differs')
            expected_score = bm['data_loss']
        parent_score = expected_score
        winner_sha = file_sha256(root/'checkpoints/initial/network.npz')
        for path in sorted(root.glob('validation-*.json'), key=lambda p:int(p.stem.split('-')[-1])):
            row = document(path);model = read(row['model'], row['sha256']);targets = row['metrics']['targets']
            metrics = panel(path.stem, targets, 'val', row.get('levels', (1,2)), final=own);values = saved_arrays(model.parent/'coordinates.npz')['offsets']
            score = metrics['data_loss']+prior(values);chosen = bool(metrics['passed'] and score < expected_score and (not own or metrics['data_loss'] < parent_score))
            if metrics != row['metrics'] or score != row['score'] or chosen != row['selected'] or expected_score != row['previous_score']:
                raise ValueError('Validation selection arithmetic differs')
            if chosen:expected_score = score;winner_sha = file_sha256(model)
        selection_path = root/'selection.json'
        if selection_path.exists():
            selection = document(selection_path);chosen = read(selection['selected'], selection['sha256'])
            # np.savez uses a new archive container: compare arrays to the identified winning source, not zip bytes.
            if selection['selected_validation_loss'] != expected_score or file_sha256(selection['source']) != winner_sha:
                raise ValueError('Selected checkpoint is not validation winner')
            source = saved_arrays(selection['source']);candidate = saved_arrays(chosen)
            if any(not np.array_equal(source[k], candidate[k]) for k in source):raise ValueError('Selection copy differs')
        else:unavailable.append('No finalized selection')
        if summary_path.exists():
            summary = document(summary_path);all_results = [r for name,r in results.items() if name.startswith('final-')]
            final, matrices = analyze_results(all_results, m, statistics, ids, m['targets'], [0,1,2], 'val', reference=False, final=True)
            if final != summary['final_metrics']:raise ValueError('Final gates differ')
            auto = document(root/'autonomy.json');parent, selected = auto['parent'], auto['selected']
            autonomy = all(preserved(parent['rhythms'][l][j], selected['rhythms'][l][j]) for l in range(2) for j in range(2))
            parent_rows = [r for name,r in results.items() if name.startswith('fresh-parent-validation-')]
            targets = summary['fresh_parent_validation']['targets']
            parent_metrics, _ = analyze_results(parent_rows, m, statistics, ids, targets, [2], 'val', reference=False)
            scores = {split:loss(matrices[2], m['targets'], ids, statistics, split) for split in ('train','val','test')}
            values = saved_arrays(chosen.parent/'coordinates.npz')['offsets'];params = codec.audit(saved_arrays(chosen), values)
            reproduced = bool(not own or abs(final['data_loss']-(expected_score-prior(values))) <= 1e-10)
            if own and reproduced != summary['selection_reproduced']:raise ValueError('Final selection reproducibility differs')
            qualified = bool(reproduced and final['passed'] and parent_metrics['passed'] and autonomy and params['passed'] and params['changed_coordinates'] > 0 and
                expected_score < selection['parent_validation_loss'] and scores['val'] < parent_metrics['data_loss'])
            if scores != summary['scores'] or params != summary['parameters'] or parent_metrics != summary['fresh_parent_validation'] or autonomy != summary['autonomy_passed'] or qualified != summary['fit_qualified_before_audit']:
                raise ValueError('Final scientific score/qualification arithmetic differs')
        else:unavailable.append('No final full-atlas/own-state validation summary')
        states = sorted((root/'optimizer').glob('*'))
        if not states:unavailable.append('No optimizer/RNG state')
        for folder in states:
            read(folder/'adam.npz');document(folder/'rng.json');document(folder/'selection.json')
        # Pin the complete small record set, exact source/environment and all model/optimizer artifacts.
        for pattern in ('run_records/**/*', 'workers/*/run_records/**/*', 'rounds/**/*', 'optimizer/**/*'):
            for path in root.glob(pattern):
                if path.is_file():read(path)
        for name in ('manifest.json','deadline.json','selection.json','history.json','source-replay.json','final-matrices.npz'):
            if (root/name).exists():read(root/name)
        if (root/'failure.json').exists():problems.append('Coordinator failed: '+str(document(root/'failure.json')))
    except Exception as exc:
        traceback.print_exc();problems.append(repr(exc))
    complete = summary_path.exists() and not unavailable
    result = dict(experiment=experiment, audit_complete=True, execution_complete=bool(complete), integrity_passed=not problems,
        problems=problems, unavailable_evidence=unavailable, candidate_parameter_checks=model_checks,
        signal_and_control_checks=task_checks, evidence_sha256=hashes,
        behavioral_gates=dict(sensory_start_stop=False, swimming=False, body_simulated=False, biological_fidelity=False),
        limits=['Saved-data audit: no independent forward rerun.',
               'Duplicate voltage maximum is independently checked against saved reduction, not unavailable full duplicate traces.',
               'All full-state phase anchors and RMD sham traces retained; no full-brain candidate trajectory archives.',
               'Optimizer/RNG state and every proposal retained; live-worker-queue exact resume not supported.',
               'Final budget receipt is written by outer supervisor after audit; qualification requires it too.'])
    write(root/'audit.json', result)
    print(json.dumps({k:result[k] for k in ('audit_complete','execution_complete','integrity_passed','problems','unavailable_evidence')}), flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run', type=Path, required=True);a=p.parse_args()
    result=audit(a.run);raise SystemExit(0 if result['audit_complete'] else 2)


if __name__ == '__main__':main()
