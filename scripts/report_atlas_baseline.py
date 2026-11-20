"""Print the E100 atlas-baseline summary compactly. Usage: python -m scripts.report_atlas_baseline --run RUN"""
import argparse
import json
from pathlib import Path


def row(label, scores):
    keys = ('pairs', 'spearman_mean_z', 'auc_excited', 'auc_inhibited', 'sign_agreement_among_significant_active',
            'model_silent_fraction_among_significant', 'permutation_p_spearman', 'permutation_p_auc_excited')
    cells = [f"{scores.get(k, float('nan')):.3g}" if isinstance(scores.get(k), (int, float)) else 'n/a' for k in keys]
    return f"{label:>22} " + ' '.join(f'{c:>9}' for c in cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    summary = json.loads((parser.parse_args().run / 'summary.json').read_text())
    print('data', summary['data'])
    print('checkpoint', summary['checkpoint_sha256'], 'chosen conductance nS', summary.get('chosen_conductance_nS'),
          'elapsed min', round(summary.get('elapsed_minutes', 0), 1), 'deadline reached', summary.get('deadline_reached'))
    print(' ' * 23 + ' '.join(f'{h:>9}' for h in ('pairs', 'rho', 'auc_exc', 'auc_inh', 'sign_ok', 'silent', 'p_rho', 'p_auc')))
    for stage in summary['stages']:
        name = stage['stage']
        if name in ('coarse_sweep', 'fine_sweep'):
            for g, tr, te in zip(stage['conductance_nS'], stage['train'], stage['test']):
                print(row(f'{name[:4]} {g:.3g} train', tr))
                print(row(f'{name[:4]} {g:.3g} test', te))
            if 'self_response' in stage:
                print('self-response fraction >0:', [round(x, 2) for x in stage['self_response']])
        elif name == 'phase':
            print(row(f"phase {stage['warmup_s']:.0f}s test", stage['test']))
        elif name == 'phase_averaged':
            print(row('phase-avg train', stage['train']))
            print(row('phase-avg test', stage['test']))
            print('self-response fraction >0:', stage.get('self_response'))
        else:
            print(name, {k: v for k, v in stage.items() if k != 'stage'})


if __name__ == '__main__':
    main()
