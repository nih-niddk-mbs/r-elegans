"""Read-only CV/peptide pilot evidence or detailed validation failure report."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def validation_failures(summary):
    """Expose the existing E123 predicates; no new thresholds or qualification."""
    failed=[];unavailable=[]
    def check(name,passed,details):
        if passed is None:unavailable.append(name)
        elif not passed:failed.append(dict(gate=name,details=details))
    value=summary.get('e122_response_reproduction_max_difference')
    check('E122 endpoint reproduction',None if value is None else math.isfinite(value) and value<=1e-10,
        dict(maximum_difference=value,tolerance=1e-10))
    for key,count in [('grids',3),('response_comparisons',2),('long',2),('long_cycle_checks',2)]:
        if len(summary.get(key,[]))!=count:unavailable.append(key+' expected '+str(count)+' entries')
    for row in summary.get('grids',[]):
        label='grid'+str(row['level'])
        check(label+' finite',row.get('finite'),row)
        error=row.get('duplicate_sham_error')
        check(label+' duplicate sham',None if error is None else math.isfinite(error) and error<1e-12,dict(error=error,tolerance=1e-12))
        for i,r in enumerate(row.get('rhythms',[])):
            check(label+f' phase{i//2} '+('RMDL' if i%2==0 else 'RMDR'),r.get('passed'),r)
        if len(row.get('rhythms',[]))!=8:unavailable.append(label+' expected8 RMD phase checks')
    for i,row in enumerate(summary.get('response_comparisons',[])):
        check(f'response grid{i}->{i+1}',row.get('passed'),row)
    restart=summary.get('restart',{})
    check('fresh-process state continuation',restart.get('passed'),restart)
    for row in summary.get('long',[]):
        label='300s grid'+str(row['level'])
        check(label+' finite',row.get('finite'),dict(finite=row.get('finite')))
        check(label+' all-cell voltage bounds',row.get('all_cell_voltage_bounded'),row)
        for i,r in enumerate(row.get('checks',[])):
            check(label+' '+('RMDL' if i==0 else 'RMDR'),r.get('passed'),r)
        if len(row.get('checks',[]))!=2:unavailable.append(label+' expected2 RMD checks')
    for i,r in enumerate(summary.get('long_cycle_checks',[])):
        check('300s coarse/fine cycle '+('RMDL' if i==0 else 'RMDR'),r.get('passed'),r)
    if not failed and not unavailable and summary.get('checks_passed') is False:
        unavailable.append('Summary flag disagrees with displayed component predicates; inspect original audit/source')
    return dict(reported_checks_passed=summary.get('checks_passed'),failed=failed,unavailable=unavailable)


def report_validation(root):
    print('READ-ONLY VALIDATION DETAILS — no simulation, fit, resume or file writes')
    print('Run:',root)
    for arm in ('network','peptide'):
        path=root/arm/'summary.json'
        if not path.exists():
            print(arm,'Summary unavailable; no work started');continue
        s=json.loads(path.read_text());print(arm,json.dumps(validation_failures(s)))
        print(arm,'Long-cycle comparisons:',json.dumps(s.get('long_cycle_checks')))
        batches=s.get('batching',[])
        if batches:
            fastest=max(batches,key=lambda r:r['target_trials_per_second'])
            print(arm,'Best measured replay batch:',fastest)
    for name in ('audit.json','budget_receipt.json'):
        path=root/name;r=json.loads(path.read_text()) if path.exists() else {}
        print(name,{k:v for k,v in r.items() if k!='artifact_sha256'})


def report(root):
    def load(name):return json.loads((root/name).read_text())
    m=load('manifest.json')
    if m['experiment'] in ('E123','E123-continuation'):return report_validation(root)
    if m['experiment']!='E122':raise ValueError('E122 pilot required')
    a=load('audit.json');b=load('budget_receipt.json');f=load('fold.json')
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
