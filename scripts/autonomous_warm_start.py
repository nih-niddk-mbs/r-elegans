"""Validate/copy a completed parameter search without changing its anchor.

    Standard library only: supervisor preflight must not initialize JAX.
"""
import json
import math
from pathlib import Path
import shutil

from scripts.experiment_records import file_sha256
from scripts.run_embodied_diagnostics import verify_input_bundle


def read_warm_start(worker):
    root=Path(worker).expanduser().resolve()
    required=('summary.json','fit_policy.json','search_state.json','best.json','network_best.npz','allowed_parameters.npz')
    for name in required:
        if not (root/name).is_file():raise ValueError('Missing warm-start record: '+str(root/name))
    summary=json.loads((root/'summary.json').read_text());policy=json.loads((root/'fit_policy.json').read_text())
    state=json.loads((root/'search_state.json').read_text());best=json.loads((root/'best.json').read_text())
    if summary.get('status')!='completed':raise ValueError('Warm-start source must be a completed, immutable worker')
    if summary.get('objective') not in ('body-wave','body-amplitude') or summary.get('fit_scope')!='neurons-connections':
        raise ValueError('Warm-start source must be a joint body recruitment search')
    if policy.get('fit_scope')!='neurons-connections':raise ValueError('Parent scope mismatch')
    factor=float(policy['maximum_density_factor'])
    if not math.isfinite(factor) or factor<=1:raise ValueError('Invalid parent bound')
    offsets=state['best']
    if len(offsets)!=summary['total_trainable'] or not all(math.isfinite(x) and abs(x)<=math.log(factor)+1e-10 for x in offsets):
        raise ValueError('Parent offsets violate original bounds/dimensions')
    if (state['completed_iterations']!=summary['iterations'] or state['selected_iteration']!=summary['selected_iteration']
            or best['iteration']!=summary['selected_iteration'] or offsets!=best['offsets']):
        raise ValueError('Parent state/selection records disagree')
    if file_sha256(root/'network_best.npz')!=summary['network_best_sha256']:raise ValueError('Parent best checksum mismatch')
    manifest=verify_input_bundle(root/'source_inputs')
    return dict(parent_path=str(root),summary=summary,policy=policy,state=state,maximum_factor=factor,
        parameters_at_bound=sum(abs(x)>=math.log(factor)-1e-8 for x in offsets),
        anchor_manifest=manifest,essential_sha256={name:file_sha256(root/name) for name in required})


def copy_warm_start(worker,destination):
    before=read_warm_start(worker);root=Path(before['parent_path']);dest=Path(destination).resolve()
    if dest==root or root in dest.parents:raise ValueError('Copy must be outside parent worker')
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Warm-start archive must not contain symlinks')
    shutil.copytree(root,dest)
    after=read_warm_start(dest)
    if before['essential_sha256']!=after['essential_sha256'] or before['anchor_manifest']!=after['anchor_manifest']:
        raise ValueError('Warm-start copy verification failed')
    # All preceding selections, traces, source/env and history are retained, not
    # merely the last genotype. This copied lineage is a local archive, not an
    # independently backed-up machine or verified cloud synchronization.
    return before
