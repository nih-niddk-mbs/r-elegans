from types import SimpleNamespace
import numpy as np
import pytest

from r_elegans.brain.autonomous_curriculum import (
    body_wave_metrics,cycle_averaged_progress,wave_refinement_agrees,cycle_movement_metrics,cycle_movement_achieved,
)


def fixture(kind='travel',speed=0.,dt=.002,frequency=4.):
    t=np.arange(0,30,dt);omega=2*np.pi*frequency;spatial=np.linspace(0,np.pi,11)
    if kind=='travel':q=.02*np.sin(omega*t[:,None]-spatial)
    elif kind=='standing':q=.02*np.sin(omega*t[:,None])*np.cos(spatial)
    elif kind=='synchronous':q=.02*np.tile(np.sin(omega*t[:,None]),(1,11))
    elif kind=='head':
        q=np.zeros((len(t),11));q[:,-3:]=.02*np.sin(omega*t[:,None])
    else:q=np.zeros((len(t),11))
    activation=q.copy()
    # A graded non-RMD oscillator is sufficient; no identity/spike requirement.
    v=np.column_stack((np.full_like(t,-50),np.full_like(t,-50),-50+.1*np.sin(omega*t)))
    m=np.full((len(t),95),.5);p=np.column_stack((speed*t+.003*np.sin(omega*t+.3),np.zeros_like(t)))
    body=SimpleNamespace(joint_angles=q,muscle_activation=activation,position=p,heading=np.zeros_like(t))
    return v,m,body,dt,1.,('RMDL','RMDR','DB01'),.75


def test_coverage_and_phase_are_separate_milestones_without_external_drive():
    head=body_wave_metrics(*fixture('head'))
    standing=body_wave_metrics(*fixture('standing'))
    synchronous=body_wave_metrics(*fixture('synchronous'))
    travel=body_wave_metrics(*fixture())
    assert head['valid'] and not head['wave_qualified'] and head['curriculum_stage']=='spread'
    assert synchronous['curriculum_stage']=='coordinate' and not synchronous['wave_qualified']
    assert not standing['wave_qualified']  # Opposite standing lobes are not a traveling wave.
    assert travel['wave_qualified'] and travel['sustained_neural_rhythms']==1
    assert travel['rmd']['RMDL']['passed'] is False
    assert travel['score']>synchronous['score']>head['score']
    assert wave_refinement_agrees(travel,body_wave_metrics(*fixture(dt=.001)))
    assert not body_wave_metrics(*fixture('static'))['wave_qualified']
    slow=body_wave_metrics(*fixture(frequency=.8))
    assert slow['wave_qualified'] and slow['frequency_hz']==pytest.approx(.8,abs=.01)
    inconsistent=list(fixture())
    inconsistent[2].muscle_activation[:]=inconsistent[2].joint_angles[:,[0]]
    assert not body_wave_metrics(*inconsistent)['wave_qualified']


def test_complete_cycle_position_averages_remove_phase_endpoint_reward():
    dt=.002;t=np.arange(0,30,dt);period=5./23.05
    q=.02*np.tile(np.sin(2*np.pi*t[:,None]/period),(1,11));h=np.zeros(len(t))
    # Equal-angle chain: mean tail-to-head axis lies halfway across its arc.
    angle=5.5*q[len(q)//2,0];axis=np.array([np.cos(angle),np.sin(angle)])
    for initial_phase in (0.,.9,2.):
        oscillation=.003*np.sin(2*np.pi*t/period+initial_phase)
        for velocity in (0.,.001):
            p=(velocity*t+oscillation)[:,None]*axis
            r=cycle_averaged_progress(q,p,h,dt,1.)
            assert r['cycle_progress_valid']
            assert len(r['phase_origins'])==2
            assert r['mean_cycle_speed_body_lengths_s']==pytest.approx(velocity,abs=2e-7)
            assert r['minimum_forward_speed_body_lengths_s']==pytest.approx(velocity,abs=2e-7)
            assert r['phase_origin_speed_spread_body_lengths_s']<2e-7


def test_drifting_body_and_passive_translation_do_not_replace_intrinsic_wave():
    values=list(fixture('head'));body=values[2];t=np.arange(len(body.joint_angles))*values[3]
    body.joint_angles[:,:7]=.001*t[:,None]
    body.muscle_activation[:,:7]=.001*t[:,None]
    assert not body_wave_metrics(*values)['wave_qualified']
    assert not cycle_movement_metrics(*fixture('static',speed=.001))['valid']
    assert not cycle_movement_metrics(*fixture('head',speed=.001))['valid']
    moving=cycle_movement_metrics(*fixture(speed=.001))
    assert cycle_movement_achieved(moving,1e-4,0.)
