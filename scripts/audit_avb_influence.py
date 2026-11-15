"""E079: read-only saved E078 current/voltage audit; no JAX or forward simulation."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.experiment_records import capture_run_record,file_sha256
from r_elegans.avb_audit import load_saved_network,analyze_pair,compare_models,density_group_members


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    for root in (repo,run):
        if out==root or root in out.parents or out in root.parents:raise ValueError('Separate external output required')
    summary=json.loads((run/'summary.json').read_text())
    if summary.get('experiment')!='E078' or not summary.get('validation_complete'):raise ValueError('Completed E078 required')
    ordinary=next(r for r in summary['validation'] if r['variant']=='ordinary' and r['grid']=='ordinary')
    selected_hash=summary['selected']['checkpoint_sha256']
    if ordinary['checkpoint_sha256']!=selected_hash:raise ValueError('Selected validation mismatch')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,a)
    evidence={str(run/'summary.json'):file_sha256(run/'summary.json')};models={};reports={}
    for label,record in [('parent',summary['baseline']),('selected',ordinary)]:
        folder=Path(record['output']).resolve()
        if run not in folder.parents:raise ValueError('Evaluation must be inside pinned source run')
        model_path=folder/'network.npz'
        if file_sha256(model_path)!=record['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
        evidence[str(model_path)]=record['checkpoint_sha256'];models[label]=load_saved_network(model_path)
        traces={}
        for condition in ('sham','inhibited'):
            path=folder/(condition+'.npz');sha=file_sha256(path)
            if sha!=record['trajectory_sha256'][condition]:raise ValueError('Saved trajectory hash mismatch')
            evidence[str(path)]=sha
            with np.load(path,allow_pickle=False) as z:
                traces[condition]={k:z[k] for k in ('dense_time_s','dense_voltage_mV')}
        report=analyze_pair(traces['sham'],traces['inhibited'],models[label]);write(out/(label+'.json'),report);reports[label]=report
        off=report['windows']['off']['cells']
        forward=[r for r in off if r['neuron'].startswith(('DB','VB'))]
        print(json.dumps(dict(model=label,AVB=[r for r in off if r['neuron'] in ('AVBL','AVBR')],
            forward_motor_neurons=len(forward),median_forward_voltage_std_ratio=float(np.median([r['voltage_std_ratio'] for r in forward if r['voltage_std_ratio'] is not None])) if any(r['voltage_std_ratio'] is not None for r in forward) else None)),flush=True)
        del traces
    changes=compare_models(models['parent'],models['selected']);write(out/'parameter_comparison.json',changes)
    schema_path=run/'parameter_groups.json'
    schema=density_group_members(json.loads(schema_path.read_text()),models['parent'])
    write(out/'density_group_inventory.json',schema);evidence[str(schema_path)]=file_sha256(schema_path)
    compact={}
    for label,r in reports.items():
        compact[label]=[{k:cell[k] for k in ('neuron','voltage_std_ratio','maximum_voltage_difference_mV','avb_L1_fraction','avb_direct_total_delta_rms_pA','incoming_edge_L1_mean_pA')}
                        for cell in r['windows']['off']['cells'] if cell['neuron'].startswith(('DB','VB'))]
    write(out/'evidence_hashes.json',evidence)
    write(out/'summary.json',dict(experiment='E079',status='completed',source_run=str(run),checkpoint_sha256={k:summary['baseline' if k=='parent' else 'selected']['checkpoint_sha256'] for k in models},
        AS_density_group_members=[r for r in schema if r['group'].startswith('density:AS:')],
        forward_motor_off_window=compact,no_training=True,no_new_simulation=True,biological_fidelity_qualified=False,
        interpretation='Sampled synapse-current reconstruction, not exact ionic balance or causal isolation. Positive current inward. AVB direct deltas hold postsynaptic voltage at sham. Do not infer intrinsic autonomous oscillation from persistent connected activity.',
        limitations='No gate states in saved traces: intrinsic channel currents not reconstructed; sensory conductances not included in synapse current totals. Sparse sampled analysis is descriptive and may miss faster events.'))
    print('Report:',out/'summary.json',flush=True)


if __name__=='__main__':main()
