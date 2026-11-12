"""Validate completed E055 state before continuing the same bounded objective."""
import json
from pathlib import Path
import numpy as np
from scripts.experiment_records import file_sha256

MATCH_KEYS=('stage','parent_checkpoint_sha256','original_anchor_sha256','readout_sha256',
    'maximum_factor','protocol','base_sensory_config','sensory_gain_order','world',
    'sensory_gain_bounds','trainable','fixed','medium')
MATCH_ARGUMENTS=('sigma','bend_target_rad','dt','brain_step_ms')


def read_completed(root,settings,size):
    root=Path(root).resolve()
    cfg=json.loads((root/'configuration.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    if summary.get('experiment')!='E055' or summary.get('status')!='completed':
        raise ValueError('Completed E055 search required')
    if file_sha256(root/'network_best.npz')!=summary['checkpoint_sha256']:
        raise ValueError('Resume network checksum mismatch')
    for key in MATCH_KEYS:
        if cfg[key]!=settings[key]:raise ValueError('Resume configuration mismatch: '+key)
    for key in MATCH_ARGUMENTS:
        if cfg['arguments'][key]!=settings['arguments'][key]:
            raise ValueError('Resume objective/numerics mismatch: '+key)
    if cfg['arguments'].get('training_cues','nominal')!=settings['arguments'].get('training_cues','nominal'):
        raise ValueError('Resume objective/numerics mismatch: training_cues; use a fresh-stage warm start')
    for key,default in (('precision','float32'),('training_margin',1.)):
        if cfg['arguments'].get(key,default)!=settings['arguments'].get(key,default):
            raise ValueError('Resume objective/numerics mismatch: '+key)
    if cfg['arguments'].get('precision','float32')=='float64':
        if cfg.get('parameterization')!='operational-parent-v1':raise ValueError('Unknown float64 parameterization')
        for name,key in (('parameter_reference.npz','parameter_reference_sha256'),('parameter_origin.npz','parameter_origin_sha256')):
            if file_sha256(root/name)!=cfg[key]:raise ValueError('Operational reference checksum mismatch')
        with np.load(root/'parameter_origin.npz') as z:
            origin=z['offsets']
            if origin.shape!=(size,) or not np.isfinite(origin).all() or np.max(np.abs(origin),initial=0)>np.log(cfg['maximum_factor'])+1e-10:
                raise ValueError('Operational reference coordinate origin invalid')
    with np.load(root/'search_state.npz',allow_pickle=False) as z:
        x=z['offsets'].copy();s=z['sensory_offsets'].copy();iteration=int(z['iteration'])
    if (x.shape!=(size,) or s.shape!=(3,) or not np.isfinite(x).all() or not np.isfinite(s).all()
            or iteration!=summary.get('last_iteration',summary['iterations'])):
        raise ValueError('Resume search state mismatch')
    if np.max(np.abs(x),initial=0)>np.log(cfg['maximum_factor'])+1e-10 or np.max(np.abs(s))>np.log(2)+1e-10:
        raise ValueError('Resume offsets exceed original bounds')
    best=json.loads((root/'best.json').read_text())
    with np.load(root/'best_state.npz',allow_pickle=False) as z:
        if (not np.array_equal(x,z['offsets']) or not np.array_equal(s,z['sensory_offsets'])
                or int(z['iteration'])!=best['iteration'] or best['iteration']!=summary['selected_iteration']):
            raise ValueError('Selected/final search states disagree')
    sensory=json.loads((root/'sensory_parameters.json').read_text())
    gains=np.exp(s)
    if (sensory['base']!=cfg['base_sensory_config'] or
            not np.array_equal(gains,sensory['gain_multipliers']) or
            not np.array_equal(gains,summary['sensory_gain_multipliers']) or
            not np.array_equal(gains,best['sensory_gain_multipliers'])):
        raise ValueError('Saved sensory parameters disagree')
    for key,i in (('chemical_gain_nS',0),('b_bend_gain_nS',1),('dva_gain_nS',2)):
        if sensory[key]!=float(sensory['base'][key]*gains[i]):
            raise ValueError('Saved effective sensory gain disagrees: '+key)
    if best['result']!=summary['best']:raise ValueError('Selected metrics disagree')
    rng=np.random.default_rng()
    rng.bit_generator.state=json.loads((root/'rng_state.json').read_text())
    return x,s,iteration,rng,summary


def latest_completed(base,pattern='sensorimotor-*'):
    roots=sorted(p for p in Path(base).glob(pattern) if p.is_dir() and (p/'summary.json').exists())
    if not roots:raise ValueError('No completed sensory-motor search found')
    root=roots[-1];cfg=json.loads((root/'configuration.json').read_text())
    size=sum(cfg['trainable'].values())-cfg['trainable']['sensory_gains']
    read_completed(root,cfg,size)
    return root
