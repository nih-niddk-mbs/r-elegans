"""E092 workers explicitly permit one documented prior exception."""
from pathlib import Path
import json
import numpy as np
from scripts import rial_sensory_worker as worker
from r_elegans.relay_cap_experiments import upper_for

_BASE=None
_PRIOR=None
_SCHEMAS=None


def initialize(inputs,schemas,cpus,records):
    global _BASE,_PRIOR,_SCHEMAS
    worker.initialize(inputs,schemas['original'],cpus,records)
    _BASE=worker._DATA.copy();_SCHEMAS=schemas
    with np.load(Path(inputs)/'cap_prior.npz',allow_pickle=False) as z:
        _PRIOR={k:z[k] for k in z.files}
    if not np.array_equal(_BASE['origin'],_PRIOR['origin']):raise ValueError('Parent-coordinate mismatch')


def evaluate(job):
    from r_elegans.brain.joint_autonomous_fit import apply_joint_offsets
    cap=job['cap'];upper=upper_for(cap,_PRIOR['upper'],int(_PRIOR['edge']))
    d=_BASE.copy()
    def decode(x):
        if x.shape!=d['origin'].shape or not np.isfinite(x).all() or np.any(x<_PRIOR['lower']-1e-12) or np.any(x>upper+1e-12):
            raise ValueError('Outside explicit experiment prior')
        return apply_joint_offsets(_BASE['base'],x-_BASE['origin'],_BASE['spec'])
    d.update(groups=_SCHEMAS[cap],lower=_PRIOR['lower'],upper=upper,decode=decode)
    worker._DATA=d
    result=worker.evaluate(job)
    result.update(cap=cap,kind=job['kind'])
    (Path(job['output'])/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
