"""Diagnostic removal of rhythmic NMJ output while preserving baseline release."""
import numpy as np
from .motor_identity import output_group

CASES=('sham','B','head','other-body','other','all')
SPLIT_CASES=('sham','RMD','SMD','SMB','head','all')


def targets_output(name,group):
    if group not in set(CASES+SPLIT_CASES):raise ValueError('Unknown NMJ diagnostic group')
    if group=='sham':return False
    if group in ('RMD','SMD','SMB'):return name.startswith(group)
    return group=='all' or output_group(name)==group


def protocol_settings(protocol):
    if protocol=='population':return 'E080',CASES,('B','head','other-body','other')
    if protocol=='head-split':return 'E081',SPLIT_CASES,('RMD','SMD','SMB')
    raise ValueError('Unknown NMJ source protocol')


def refinement_target(comparisons,protocol):
    _,_,eligible=protocol_settings(protocol)
    rows=[r for r in comparisons if r['case'] in eligible and r['metrics']['pre_cue_match'] and r['metrics']['safety_passed']]
    return min(rows,key=lambda r:r['metrics']['during_to_sham_raw_range_ratio'])['case'] if rows else None


def split_controls(comparisons):
    rows={r['case']:r['metrics'] for r in comparisons}
    head=rows.get('head');all_output=rows.get('all')
    quality=lambda r:r is not None and r['pre_cue_match'] and r['safety_passed'] and r['recovery_amplitude_preserved']
    head_ok=bool(quality(head) and abs(head['during_to_sham_raw_range_ratio']-.11378497784610712)<=.05)
    all_ok=bool(quality(all_output) and all_output['during_to_sham_raw_range_ratio']<=.01)
    return dict(combined_head_reproduces_E080=head_ok,all_output_static_control=all_ok,
                passed=head_ok and all_ok,scope='Prospective diagnostic control checks, not learned stopping')


def release_hold(ids,threshold,slope,baseline,group):
    import jax.numpy as jnp
    import jax
    if group not in set(CASES+SPLIT_CASES):raise ValueError('Unknown NMJ diagnostic group')
    mean=np.asarray(baseline)
    if mean.shape!=(len(ids),) or not np.isfinite(mean).all() or np.any((mean<0)|(mean>1)):
        raise ValueError('Baseline releases must be probabilities per neuron')
    if np.any(np.asarray(slope)<=0):raise ValueError('Positive release slopes required')
    mask=jnp.asarray([targets_output(n,group) for n in ids])
    mean=jnp.asarray(mean)
    def apply(voltage,t):
        live=jax.nn.sigmoid((voltage-threshold)/slope)
        return jnp.where(mask & (t>=30.) & (t<40.),mean,live)
    return apply
