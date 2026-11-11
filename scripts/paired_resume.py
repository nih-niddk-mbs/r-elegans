"""Validate a completed paired-search continuation; preserve its original anchor."""
import json
from pathlib import Path

from scripts.experiment_records import file_sha256

MATCH_ARGUMENTS=('uids','frames','start_frame','media','dt','brain_step_ms','burn_in',
                 'maximum_factor','sigma','basal_brightness','sampling')


def read_resume(root,settings,size):
    import numpy as np
    root=Path(root).resolve()
    summary=json.loads((root/'summary.json').read_text())
    config=json.loads((root/'configuration.json').read_text())
    if summary.get('experiment')!='E050' or summary.get('status')!='completed':
        raise ValueError('Resume requires a completed E050 paired search')
    if file_sha256(root/'network_best.npz')!=summary['checkpoint_sha256']:
        raise ValueError('Resume checkpoint checksum mismatch')
    for key in ('network_sha256','readout_sha256','prepared_manifest_sha256','split_sha256','network_config','trainable'):
        if config[key]!=settings[key]:raise ValueError('Resume configuration mismatch: '+key)
    for key in MATCH_ARGUMENTS:
        if config['arguments'][key]!=settings['arguments'][key]:
            raise ValueError('Resume objective/numerics mismatch: '+key)
    with np.load(root/'search_state.npz',allow_pickle=False) as z:
        offsets=np.asarray(z['offsets']).copy();iteration=int(z['iteration'])
    expected_iteration=summary.get('last_iteration',summary['iterations'])
    if iteration!=expected_iteration or offsets.shape!=(size,) or not np.isfinite(offsets).all():
        raise ValueError('Resume search state mismatch')
    if np.max(np.abs(offsets),initial=0)>np.log(settings['arguments']['maximum_factor'])+1e-10:
        raise ValueError('Resume offsets exceed original anchor bounds')
    rng=np.random.default_rng()
    rng.bit_generator.state=json.loads((root/'rng_state.json').read_text())
    return offsets,iteration,rng,summary


def latest_completed(base):
    roots=sorted(p for p in Path(base).glob('paired-fit-*') if p.is_dir() and (p/'summary.json').is_file())
    if not roots:raise ValueError('No completed paired fit found')
    root=roots[-1]
    s=json.loads((root/'summary.json').read_text())
    if s.get('status')!='completed' or s.get('experiment')!='E050':
        raise ValueError('Latest paired summary is not a completed E050 run')
    if file_sha256(root/'network_best.npz')!=s['checkpoint_sha256']:
        raise ValueError('Latest checkpoint checksum mismatch')
    for name in ('search_state.npz','rng_state.json','configuration.json'):
        if not (root/name).is_file():raise ValueError('Missing resume artifact: '+name)
    return root
