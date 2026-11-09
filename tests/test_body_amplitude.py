from types import SimpleNamespace
import numpy as np

from r_elegans.brain.body_amplitude import body_amplitude_metrics,amplitude_refinement_agrees


def example(pattern='synchronous',dt=.002):
    t=np.arange(0,30,dt);phase=np.zeros(11)
    if pattern=='opposed':phase=np.arange(11)*np.pi
    frequency=np.linspace(.8,4.,11) if pattern=='asynchronous' else np.full(11,3.)
    q=.002*np.sin(2*np.pi*t[:,None]*frequency+phase)
    if pattern=='head':q[:,:8]=0.
    elif pattern=='static':q[:]=.2
    elif pattern=='ramp':q[:]=t[:,None]*.001
    elif pattern=='decay':q*=np.exp(-t[:,None]/2.)
    body=SimpleNamespace(joint_angles=q,muscle_activation=q.copy())
    v=-50+.2*np.tile(np.sin(2*np.pi*t[:,None]),(1,3));m=np.full((len(t),95),.5)
    return v,m,body,dt,.001,('RMDL','RMDR','DB01'),.75


def test_amplitude_ignores_phase_and_frequency_relationships():
    reports=[body_amplitude_metrics(*example(kind)) for kind in ('synchronous','opposed','asynchronous')]
    assert all(r['body_amplitude_qualified'] for r in reports)
    assert all(not r['wave_qualified'] and not r['locomotion_demonstrated'] for r in reports)
    # Regional mean cancels opposite phases; independent-joint amplitudes do not.
    assert reports[1]['regional_joint_median_p2p_rad'][0]>.003
    assert amplitude_refinement_agrees(reports[0],body_amplitude_metrics(*example(dt=.001)))
    passive_body=list(example())
    passive_body[2].muscle_activation[:,:8]=0.  # Posterior passive transmission is allowed.
    assert body_amplitude_metrics(*passive_body)['body_amplitude_qualified']


def test_static_ramps_transients_and_head_only_do_not_qualify():
    for kind in ('head','static','ramp','decay'):
        report=body_amplitude_metrics(*example(kind))
        assert not report['body_amplitude_qualified']
    head=body_amplitude_metrics(*example('head'))
    assert head['regional_joint_median_p2p_rad'][0]==0
    assert head['regional_joint_median_p2p_rad'][1]==0
