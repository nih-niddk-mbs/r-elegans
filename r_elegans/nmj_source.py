"""Diagnostic removal of rhythmic NMJ output while preserving baseline release."""
import numpy as np
from .motor_identity import output_group

CASES=('sham','B','head','other-body','other','all')


def release_hold(ids,threshold,slope,baseline,group):
    import jax.numpy as jnp
    import jax
    if group not in CASES:raise ValueError('Unknown NMJ diagnostic group')
    mean=np.asarray(baseline)
    if mean.shape!=(len(ids),) or not np.isfinite(mean).all() or np.any((mean<0)|(mean>1)):
        raise ValueError('Baseline releases must be probabilities per neuron')
    if np.any(np.asarray(slope)<=0):raise ValueError('Positive release slopes required')
    mask=jnp.asarray([group=='all' or output_group(n)==group for n in ids])
    mean=jnp.asarray(mean)
    def apply(voltage,t):
        live=jax.nn.sigmoid((voltage-threshold)/slope)
        return jnp.where(mask & (t>=30.) & (t<40.),mean,live)
    return apply
