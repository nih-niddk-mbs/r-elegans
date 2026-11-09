"""Recheck saved zero-input channel profiles from alternate voltage basins."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from r_elegans.brain.conductance_network import load_network
from r_elegans.brain.posterior_excitability import isolated_recording
from scripts.assay_posterior_excitability import summarize
from scripts.experiment_records import capture_run_record,file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();source=args.audit_dir.expanduser().resolve();out=args.output_dir.expanduser().resolve()
    repo=Path(__file__).resolve().parents[1]
    if out==repo or repo in out.parents or out==source or source in out.parents:parser.error('Keep output outside repo/source')
    record=json.loads((source/'summary.json').read_text())
    if record.get('experiment')!='E040' or record.get('status')!='completed':parser.error('Completed E040 audit required')
    profiles_path=source/'profiles.npz'
    if file_sha256(profiles_path)!=record['profiles_sha256']:parser.error('Profile checksum mismatch')
    anchor_path=source/'parent_worker/source_inputs/network.npz'
    if file_sha256(anchor_path)!=record['original_anchor_sha256']:parser.error('Anchor checksum mismatch')
    p,f,c=load_network(anchor_path)
    with np.load(profiles_path,allow_pickle=False) as z:
        focus=z['focus_indices'];profiles=z['original_anchor_offsets'];names=z['names'].tolist();ids=z['neuron_ids'].tolist()
    bound=np.log(record['maximum_factor'])
    if not np.isfinite(profiles).all() or np.abs(profiles).max()>bound+1e-8 or np.any(profiles[:,:,-1]!=0):
        parser.error('Invalid original-bound profiles or changed leak')
    if not np.all(np.asarray(f.bias_pA)==0):parser.error('Zero input required')
    out.mkdir(parents=True,exist_ok=False);capture_run_record(out,repo,args)
    cells=jax.tree.map(lambda x:x[focus],f.cells);anchor=np.asarray(p.log_channel_scale)[focus]
    run=jax.jit(lambda g,v:isolated_recording(cells._replace(conductances_nS=g),v,.002,6000,20))
    start=time.monotonic();rows=[]
    for initial in (-65.,-50.,-35.):
        for name,offset in zip(names,profiles):
            g=cells.conductances_nS*jnp.exp(jnp.asarray(anchor+offset,dtype=cells.conductances_nS.dtype))
            v,ca=jax.device_get(run(g,jnp.full(len(focus),initial,dtype=g.dtype)))
            m=summarize(v,ca,f.release_threshold_mV[focus],f.release_slope_mV[focus],ids)
            label=f'{name}-initial-{abs(int(initial))}'
            np.savez_compressed(out/(label+'.npz'),voltage_mV=v,calcium_uM=ca)
            rows.append(dict(profile=name,initial_voltage_mV=initial,metrics=m))
            print(json.dumps(dict(profile=name,initial_voltage_mV=initial,rhythmic_count=m['rhythmic_count'],maximum_sustained_p2p_mV=m.get('maximum_sustained_p2p_mV'))),flush=True)
            (out/'progress.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    summary=dict(experiment='E040',mode='alternate_initial_voltage_screen',status='completed',source_audit=str(source),
        source_summary_sha256=file_sha256(source/'summary.json'),profiles_sha256=file_sha256(profiles_path),
        initial_voltages_mV=[-65,-50,-35],injected_current_pA=0,seconds=12,step_ms=.1,results=rows,
        total_cell_trials=len(rows)*len(focus),elapsed_seconds=time.monotonic()-start,
        limitations=['Finite profile/initial-state/time screen, not proof of impossibility.',
                     'No synapses or body;initial voltage interventions are not sustained input.',
                     'Any newly rhythmic regime requires longer and finer-step confirmation before use.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(total_cell_trials=summary['total_cell_trials'],rhythmic_cell_trials=sum(r['metrics']['rhythmic_count'] for r in rows),elapsed_seconds=summary['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
