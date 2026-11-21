"""Compare E105 extrasynaptic-term runs side by side (read-only)."""
import argparse
import json
from pathlib import Path


def metric(block, key):
    value = (block or {}).get(key)
    return 'n/a' if value is None else f'{value:.3f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    sweep = parser.parse_args().sweep
    print(f"{'run':26s} {'net':>3s} {'steps':>6s} | {'val sp0':>7s} {'val spB':>7s} (step) | {'test sp0':>8s} {'test spB':>8s} {'aucB':>6s} | "
          f"{'pep-off spB':>11s} {'aucOff':>6s} | {'tau s':>5s} {'exc>.1':>6s} {'inh>.1':>6s} {'conv':>6s}")
    for folder in sorted(p for p in sweep.iterdir() if p.is_dir() and (p / 'summary.json').exists()):
        s = json.loads((folder / 'summary.json').read_text())
        first = s['evaluations'][0] if s['evaluations'] else {}
        best, off, pep = s['best_val_checkpoint'], s['best_val_peptides_off'], s['peptide']
        conv = (s.get('substep_convergence') or {}).get('max_abs_difference_over_max_abs_response')
        print(f"{folder.name:26s} {('yes' if s['train_network'] else 'no'):>3s} {s['steps']:6d} | {metric(first.get('val'), 'spearman_mean_z'):>7s} "
              f"{metric(best['val'], 'spearman_mean_z'):>7s} ({s['best_val']['step']}) | {metric(first.get('test'), 'spearman_mean_z'):>8s} "
              f"{metric(best['test'], 'spearman_mean_z'):>8s} {metric(best['test'], 'auc_excited'):>6s} | {metric(off['test'], 'spearman_mean_z'):>11s} "
              f"{metric(off['test'], 'auc_excited'):>6s} | {pep['tau_s']:5.1f} {pep['pairs_excitatory_above_0p1']:6d} {pep['pairs_inhibitory_above_0p1']:6d} "
              f"{'n/a' if conv is None else f'{conv:.3f}':>6s}")
    print('sp0 = first evaluation (start), spB = best-validation checkpoint, pep-off = the same checkpoint with the peptide term switched off; '
          'test animals are evaluated, never used for selection')


if __name__ == '__main__':
    main()
