"""E093 setup preserves the E092 audited one-edge prior exception explicitly."""
import json,os,multiprocessing
from pathlib import Path
import numpy as np
from scripts import rial_sensory_worker as core
from scripts.experiment_records import capture_run_record
from r_elegans.relay_cap_experiments import upper_for


def setup(inputs):
    from r_elegans.brain.conductance_network import load_network
    from r_elegans.brain.joint_autonomous_fit import apply_joint_offsets
    from r_elegans.rhythm_comparison import assert_fixed_equal
    d=core.setup(inputs)
    with np.load(inputs/'cap_prior.npz',allow_pickle=False) as z:
        origin,lower,upper,edge=z['origin'],z['lower'],z['upper'],int(z['edge'])
    allowed=upper_for('aiyr_2p5',upper,edge)
    parent,fixed,config=load_network(inputs/'motion_parent.npz');assert_fixed_equal(d['fixed'],fixed,d['config'],config)
    if np.any(origin<lower-1e-12) or np.any(origin>allowed+1e-12):raise ValueError('Motion parent exceeds explicit prior')
    rebuilt=apply_joint_offsets(d['base'],origin-d['origin'],d['spec'])
    if any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(rebuilt,parent)):raise ValueError('Motion parent reconstruction mismatch')
    def decode(x):
        if x.shape!=origin.shape or not np.isfinite(x).all() or np.any(x<lower-1e-12) or np.any(x>allowed+1e-12):raise ValueError('Motion candidate exceeds explicit prior')
        return apply_joint_offsets(parent,x-origin,d['spec'])
    d.update(base=parent,origin=origin,lower=lower,upper=allowed,decode=decode)
    return d


def initialize(inputs,schemas,cpus,records):
    slot=(multiprocessing.current_process()._identity[-1]-1)%len(cpus);os.sched_setaffinity(0,cpus[slot])
    core._DATA=setup(Path(inputs));core._DATA['groups']=schemas['aiyr_2p5']
    capture_run_record(Path(records)/f'worker-{slot}',Path(__file__).resolve().parents[1],dict(inputs=inputs,cpus=cpus[slot],experiment='E093'))


def evaluate(job):
    result=core.evaluate(job);result.update(cap='aiyr_2p5',kind=job['kind'])
    (Path(job['output'])/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
