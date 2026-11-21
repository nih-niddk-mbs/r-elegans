"""Compare E104 all-parameter runs side by side (read-only)."""
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
    print(f"{'run':22s} {'set':8s} {'steps':>6s} {'min':>5s} | {'val sp0':>7s} {'val spF':>7s} {'val spB':>7s} (best@step) | "
          f"{'test sp0':>8s} {'test spB':>8s} {'test aucB':>9s} | {'conv diff':>9s} {'extra@bound'}")
    for folder in sorted(p for p in sweep.iterdir() if p.is_dir() and (p / 'summary.json').exists()):
        s = json.loads((folder / 'summary.json').read_text())
        first = s['evaluations'][0] if s['evaluations'] else {}
        best, final = s['best_val_checkpoint'], s['final']
        convergence = (s.get('substep_convergence') or {}).get('max_abs_difference_over_max_abs_response')
        extra = s['at_clip_bound'].get('extra')
        print(f"{folder.name:22s} {s['parameter_set']:8s} {s['steps']:6d} {s['elapsed_minutes']:5.0f} | "
              f"{metric(first.get('val'), 'spearman_mean_z'):>7s} {metric(final['val'], 'spearman_mean_z'):>7s} {metric(best['val'], 'spearman_mean_z'):>7s} "
              f"({s['best_val']['step']}) | {metric(first.get('test'), 'spearman_mean_z'):>8s} {metric(best['test'], 'spearman_mean_z'):>8s} "
              f"{metric(best['test'], 'auc_excited'):>9s} | {'n/a' if convergence is None else f'{convergence:.3f}':>9s} {sum(extra.values()) if extra else 0}")
    print('sp0 = first evaluation (start network), spF = final, spB = best-validation checkpoint; test numbers are evaluated, never used for selection')


if __name__ == '__main__':
    main()
