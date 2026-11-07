"""Rebuild diagnostic inputs from a successful checkpoint already on this machine.

No download, fresh random policy, training or silent fallback. The rebuilt
candidate may differ from the Mac candidate because its parent differs; source,
selected checkpoint and resulting model hashes are retained explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def find_successful_parent(results):
    for run in sorted(Path(results).glob('head-sweep-*'),reverse=True):
        try:summary=json.loads((run/'summary.json').read_text())
        except (OSError,ValueError):continue
        for worker in sorted(summary.get('workers',[]),key=lambda w:w['seed']):
            result=worker.get('stages',{}).get('head_sweep',{})
            parent=run/f"seed-{worker['seed']}/head_sweep/actor_best.npz"
            if result.get('passed') is True and result.get('exit_code',0)==0 and parent.is_file():
                return parent
    raise ValueError('No successful full-network head-sweep checkpoint found under '+str(results)+
        '. Supply --parent-actor explicitly, or use --inputs for a transferred exact candidate.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-actor',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];output=args.output_dir.expanduser().resolve()
    if output==repo or repo in output.parents:parser.error('Keep inputs external')
    output.mkdir(parents=True,exist_ok=False)
    if __package__:
        from .experiment_records import capture_run_record
    else:
        from experiment_records import capture_run_record
    capture_run_record(output,repo,args)
    import numpy as np
    from r_elegans.model import load_builtin_model
    from r_elegans.rl.full_network_actor import actor_from_arrays
    from r_elegans.brain.conductance_network import initialize_conductance_network,save_network,ConductanceNetworkConfig
    from r_elegans.brain.rhythm_qualification import apply_rmd_rhythm_prior
    with np.load(args.parent_actor,allow_pickle=False) as data:parent=actor_from_arrays(data)
    model=load_builtin_model();config=ConductanceNetworkConfig()
    params,fixed=initialize_conductance_network(parent,model.neuron_ids,config)
    params,fixed,_=apply_rmd_rhythm_prior(params,fixed,model.neuron_ids)
    save_network(output/'network.npz',params,fixed,config)
    shutil.copy2(args.parent_actor,output/'parent_actor.npz')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(experiment='E028',initialization='rebuilt from existing successful server parent; no random fallback',
        parent_path=str(args.parent_actor.expanduser().resolve()),parent_sha256=sha(args.parent_actor),
        sha256={name:sha(output/name) for name in ('network.npz','parent_actor.npz')},
        profile='E027 exploratory RMD profile, fixed all-cell model; not a physiological fit',
        exact_mac_checkpoint_assumed=False)
    (output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
