"""Tabulate a polarity sweep: gait/recovery flags and gait changes relative to the saved E099 evaluation (read-only)."""
import argparse
import json
from pathlib import Path

import numpy as np


def gait_change(windows):
    saved = (windows.get('saved') or {}).get('no_cue')
    new = (windows.get('polarized') or {}).get('no_cue')
    if not saved or not new:
        return None, None
    mean = lambda values: float(np.mean([np.nan if v is None else v for v in values]))
    frequency = float(np.mean([mean(n['frequency_hz']) / mean(s['frequency_hz']) for n, s in zip(new, saved)]))
    cycles = np.mean([np.asarray(n['regional_cycles_rad']) / np.asarray(s['regional_cycles_rad']) for n, s in zip(new, saved)], axis=0)
    return frequency, cycles.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--extra', action='append', default=[], metavar='LABEL=comparison.json',
                        help='additional completed comparison files (e.g. the full-polarity run)')
    args = parser.parse_args()
    manifest = json.loads((args.sweep / 'variants' / 'variants.json').read_text())
    sources = {name: args.sweep / 'gait' / name / 'comparison.json' for name in manifest['variants']}
    sources.update({item.split('=', 1)[0]: Path(item.split('=', 1)[1]) for item in args.extra})
    rows = []
    for name, path in sources.items():
        info = manifest['variants'].get(name, {})
        row = dict(variant=name, edges_changed=info.get('edges_changed'), weight_fraction_changed=info.get('weight_fraction_changed'))
        if not path.exists():
            rows.append(dict(row, status='no result'))
            continue
        d = json.loads(path.read_text())
        frequency, cycles = gait_change(d['gait_windows'])
        flags = d['polarized_flags']
        rows.append(dict(row, status=d['polarized_status'], safe=flags.get('safe'), baseline_and_recovery=flags.get('baseline_and_recovery_preserved'),
                         relay_component=flags.get('relay_component_preserved'), relay_range=flags.get('relay_operating_range_passed'),
                         useful_response=flags.get('useful_response_preserved'), frequency_ratio=frequency, regional_cycle_ratio=cycles))
    (args.sweep / 'sweep_report.json').write_text(json.dumps(rows, indent=1, default=float) + '\n')
    fmt = lambda v, f='{:.3f}': 'n/a' if v is None else f.format(v)
    print(f"{'variant':20s} {'edges':>6s} {'wfrac':>6s} {'status':>10s} {'safe':>5s} {'base+rec':>8s} {'relayR':>6s} {'freqx':>6s}  regional cycle ratios")
    for r in rows:
        cyc = ' '.join(fmt(c, '{:.2f}') for c in r.get('regional_cycle_ratio') or [])
        print(f"{r['variant']:20s} {fmt(r['edges_changed'], '{}'):>6s} {fmt(r['weight_fraction_changed'], '{:.2f}'):>6s} {r['status']:>10s} "
              f"{str(r.get('safe')):>5s} {str(r.get('baseline_and_recovery')):>8s} {str(r.get('relay_range')):>6s} {fmt(r.get('frequency_ratio')):>6s}  {cyc}")


if __name__ == '__main__':
    main()
