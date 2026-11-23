"""Compact saved E122 coverage/checkpoint/resource evidence; no forward run or fit."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def report(root):
    def load(name):return json.loads((root/name).read_text())
    m=load('manifest.json');a=load('audit.json');b=load('budget_receipt.json');f=load('fold.json')
    if m['experiment']!='E122':raise ValueError('E122 pilot required')
    print('Run:',root)
    print('Receipt:',{k:b.get(k) for k in ('all_steps_finished','timed_out','audit_complete','elapsed_seconds')})
    print('Integrity:',{k:a.get(k) for k in ('integrity_passed','problems')})
    print('Coverage:',dict(usable_animals=f['usable_animals'],accepted_trials=f['accepted_trials'],
        train_animals=len(f['subjects']['train_subjects']),heldout_animals=len(f['subjects']['heldout_subjects']),
        train_observed_pairs=len(f['stats']['train']['pairs']),heldout_observed_pairs=len(f['stats']['heldout']['pairs']),
        training_targets=m['targets']))
    for arm in ('network','peptide'):
        s=load(arm+'/summary.json');r=s['selected']
        selected_path=root/arm/r['name']/'network.npz'
        print(arm,dict(selected=str(selected_path),recorded_sha256=r['sha256'][str(selected_path)],
            short_window_rmd_passed=all(c['passed'] for c in r['short_window_rmd_checks']),
            duplicate_sham_error=r['duplicate_sham_error'],parameter_audit=r['parameter_audit'],
            peptide_bounds_passed=r['peptide_bounds_passed'],peptide_log_gains=r['peptide_log_gains'],
            maximum_final_peptide_conductance_nS=r['maximum_final_peptide_conductance_nS'],
            all_evaluations_safe=a['arms'][arm]['all_evaluations_safe'],selection_history=s['updates']))
    print('Signal comparison:',a.get('signal_comparison'))
    telemetry={}
    path=root/'gpu_telemetry.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row.get('returncode')!=0:continue
            for values in csv.reader(row.get('rows',[])):
                try:index,util,memory,power=map(float,values)
                except ValueError:continue
                telemetry.setdefault(int(index),[]).append((util,memory,power))
    print('GPU samples:',{k:dict(samples=len(v),mean_utilization_percent=statistics.mean(x[0] for x in v),
        peak_memory_MiB=max(x[1] for x in v),mean_power_W=statistics.mean(x[2] for x in v)) for k,v in telemetry.items()})
    print('Retained run GiB:',round(sum(p.stat().st_size for p in root.rglob('*') if p.is_file())/1024**3,3))
    print('Unavailable evidence:',json.dumps(a.get('unavailable_evidence',[])))
    print('Saved-data report only; no new qualification or training. GPU samples include startup/idle periods.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();report(a.run.resolve())


if __name__=='__main__':main()
