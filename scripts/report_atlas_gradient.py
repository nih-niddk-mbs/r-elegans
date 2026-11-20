"""Print the post-run audit facts of an E101 run (read-only)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    magnitude = np.abs(np.asarray(values, float)).ravel()
    return {q: float(np.quantile(magnitude, p)) for q, p in (('median', .5), ('p90', .9), ('p99', .99), ('max', 1.))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    summary = json.loads((run / 'summary.json').read_text())
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines() if line]
    steps = [h for h in history if 'loss' in h and 'pooled_corr' in h]
    hashes = {name: (sha256(run / name) == summary[key]) for name, key in
              (('network_final.npz', 'final_sha256'), ('network_best_train_loss.npz', 'best_train_loss_sha256'))}
    first, last = steps[0], steps[-1]
    responses = np.load(run / 'responses.npz')
    out = dict(
        execution=dict(status=summary['status'], stop_reason=summary['stop_reason'], steps=summary['steps'], elapsed_minutes=round(summary['elapsed_minutes'], 1),
                       budget_minutes=summary['budget_minutes'], deadline_reached=summary['deadline_reached'], devices=summary['devices']),
        selection=dict(rule=summary['selection_rule'], best_train_loss=summary['best_train_loss']),
        parameters=dict(trainable=summary['trainable'], changed_entries=summary['changed_entries'],
                        frozen_parameter_changes=summary['frozen_parameter_changes'], at_clip_bound=summary['at_clip_bound'],
                        max_abs_change=summary['max_abs_change']),
        signal=dict(first_step=dict(loss=first['loss'], pooled=first['pooled_corr'], per_target=first['per_target_corr'], grad_norms=first['grad_norms']),
                    last_step=dict(loss=last['loss'], pooled=last['pooled_corr'], per_target=last['per_target_corr'], grad_norms=last['grad_norms']),
                    nonfinite_steps=sum(1 for h in history if h.get('finite') is False)),
        initial_evaluation=summary['evaluations'][0] if summary['evaluations'] else None,
        evaluation_curve=[dict(step=e['step'], train_spearman=e['train'].get('spearman_mean_z'), test_spearman=e['test'].get('spearman_mean_z'),
                               train_auc_excited=e['train'].get('auc_excited'), test_auc_excited=e['test'].get('auc_excited'),
                               test_silent=e['test'].get('model_silent_fraction_among_significant')) for e in summary['evaluations']],
        response_magnitude=dict(initial=quantiles(responses['initial']), final=quantiles(responses['final'])),
        final=summary['final'], substep_convergence=summary['substep_convergence'],
        artifacts=dict(hashes_match_summary=hashes, run_records=sorted(p.name for p in (run / 'run_records').iterdir()),
                       parent_sha256=summary['parent_sha256'], parent_copy_sha256=sha256(run / 'parent_network.npz'),
                       anchor_sha256=summary.get('anchor_sha256'), optimizer_start=summary.get('optimizer_start')),
        unavailable=summary['unavailable'])
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
