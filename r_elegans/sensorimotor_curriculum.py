"""Controlled sensory–motor prerequisites, not an ancestral reconstruction.

A cue is an environmental ligand field, not a neural/muscle target. Numerical
acceptance thresholds are engineering gates, not a measured diacetyl reflex.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ResponseProtocol:
    duration_s: float = 8.
    onset_s: float = 2.
    washout_s: float = 3.
    recovery_s: float = 6.
    bend_target_rad: float = .00003
    sensory_target_mV: float = .1
    recovery_fraction: float = .25
    maximum_joint_rad: float = .5

    def __post_init__(self):
        values=tuple(vars(self).values())
        if not all(np.isfinite(v) for v in values) or not (
            0 < self.onset_s < self.washout_s < self.recovery_s < self.duration_s
            and self.bend_target_rad>0 and self.sensory_target_mV>0
            and 0<self.recovery_fraction<1 and self.maximum_joint_rad>0):
            raise ValueError('Ordered positive response protocol required')


def cue_world(protocol=ResponseProtocol(), *, dose=1., onset_shift=0., sham=False):
    """Broad diacetyl Gaussian field with one onset and washout; no other cues.

    Width10mm makes the fixture nearly a bath over one1mm worm, while retaining
    actual head-position sampling. Dose is micromolar, not calibrated OP50 odor.
    """
    import jax.numpy as jnp
    from r_elegans.body.embodied_sensory import ChemicalWorld,validate_chemical_world
    if not np.isfinite(dose) or dose<=0 or not np.isfinite(onset_shift):
        raise ValueError('Positive finite cue dose and finite timing required')
    onset=protocol.onset_s+onset_shift;washout=protocol.washout_s+onset_shift
    if not 0<onset<washout<protocol.recovery_s:raise ValueError('Cue must precede recovery')
    world=ChemicalWorld(jnp.zeros((3,2)),jnp.full(3,.01),jnp.array([0.,0.,50000.]),
        jnp.array([0. if sham else dose,0.,0.]),jnp.array([onset]*3),jnp.array([washout]*3))
    validate_chemical_world(world)
    return world


def response_metrics(active,sham,neuron_ids,protocol=ResponseProtocol(),*,onset_shift=0.):
    """Score incremental cue response and recovery, never raw posture alone.

    Full dynamics remain reported separately. Cue-minus-sham does not assert
    linearity; both are independently simulated closed-loop physical trajectories.
    """
    t=np.asarray(active['dense_time_s'],dtype=float)
    if (t.ndim!=1 or len(t)<40 or not np.isfinite(t).all() or np.any(np.diff(t)<=0)
            or not np.array_equal(t,np.asarray(sham['dense_time_s']))
            or t[0]!=0 or t[-1]<protocol.duration_s):
        raise ValueError('Matching finite complete response grids required')
    onset=protocol.onset_s+onset_shift;washout=protocol.washout_s+onset_shift
    if not 0<onset<washout<protocol.recovery_s:raise ValueError('Invalid stimulus interval')
    q=np.asarray(active['dense_joints'],dtype=float);qs=np.asarray(sham['dense_joints'],dtype=float)
    v=np.asarray(active['dense_voltage_mV'],dtype=float);vs=np.asarray(sham['dense_voltage_mV'],dtype=float)
    if (q.shape!=(len(t),11) or qs.shape!=q.shape or v.shape!=(len(t),len(neuron_ids))
            or vs.shape!=v.shape or not all(np.isfinite(a).all() for a in (q,qs,v,vs))):
        raise ValueError('Finite matched body and neural traces required')
    pre=t<onset;response=(t>=onset)&(t<protocol.recovery_s);recovery=t>=protocol.recovery_s
    if not all(mask.any() for mask in (pre,response,recovery)):raise ValueError('Incomplete protocol windows')
    delta=q-qs
    pre_difference=float(np.max(np.abs(delta[pre])))
    delta=delta-delta[pre].mean(axis=0)
    peak=np.max(np.abs(delta[response]),axis=0)
    residual=np.max(np.abs(delta[recovery]),axis=0)
    # The best joint must satisfy BOTH response and recovery, not different joints.
    normalized_residual=residual/np.maximum(peak,1e-12)
    scores=np.minimum(peak/protocol.bend_target_rad,1.)-.5*np.maximum(
        normalized_residual-protocol.recovery_fraction,0.)
    joint=int(np.argmax(scores))
    sensory=[neuron_ids.index(n) for n in ('AWAL','AWAR')]
    dv=v[:,sensory]-vs[:,sensory]
    recruitment=float(np.max(np.mean(dv[(t>=onset)&(t<washout)],axis=1)))
    sensory_residual=float(np.max(np.abs(dv[recovery].mean(axis=1))))
    sensory_recovery_ratio=sensory_residual/max(recruitment,1e-12)
    neural_deficit=max(0.,1.-recruitment/protocol.sensory_target_mV)
    amplitude_deficit=max(0.,1.-peak[joint]/protocol.bend_target_rad)
    recovery_penalty=max(0.,normalized_residual[joint]-protocol.recovery_fraction)
    sensory_recovery_penalty=max(0.,sensory_recovery_ratio-protocol.recovery_fraction)
    maximum_joint=float(max(np.max(np.abs(q)),np.max(np.abs(qs))))
    safety_penalty=max(0.,maximum_joint/protocol.maximum_joint_rad-1.)
    pre_penalty=pre_difference/protocol.bend_target_rad
    passed=bool(peak[joint]>=protocol.bend_target_rad and
        residual[joint]<=protocol.recovery_fraction*peak[joint] and
        recruitment>=protocol.sensory_target_mV and maximum_joint<protocol.maximum_joint_rad
        and sensory_recovery_ratio<=protocol.recovery_fraction
        and pre_difference<1e-7)
    return dict(total=float(amplitude_deficit+recovery_penalty+.25*(neural_deficit+sensory_recovery_penalty)+10*safety_penalty+pre_penalty),
        passed=passed,selected_joint=joint,cue_specific_peak_rad=float(peak[joint]),
        cue_specific_recovery_residual_rad=float(residual[joint]),recovery_ratio=float(normalized_residual[joint]),
        peak_by_joint_rad=peak.tolist(),recovery_residual_by_joint_rad=residual.tolist(),
        awa_depolarization_mV=recruitment,maximum_absolute_joint_rad=maximum_joint,
        awa_recovery_residual_mV=sensory_residual,awa_recovery_ratio=float(sensory_recovery_ratio),
        pre_cue_difference_rad=pre_difference,
        scope='Engineering reversible cue-response gate; not autonomous rhythm, navigation or biological validation')
