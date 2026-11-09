from types import SimpleNamespace
import numpy as np
from r_elegans.brain.posterior_motor import (
    MotorTargets,posterior_motor_targets,posterior_motor_metrics,posterior_motor_refinement_agrees,
    rhythmic_release_sources,chemical_driver_paths,posterior_motor_proposal,
)
from r_elegans.brain.joint_autonomous_fit import JointFitSpec
from r_elegans.body.neuromuscular import build_muscle_projection,muscle_longitudinal_positions,BODY_WALL_MUSCLE_NAMES


def test_target_selection_excludes_nonmotor_and_anterior_motor_projections():
    d=np.zeros((11,4));d[:4,:3]=1/3;d[4:,3]=1
    t=posterior_motor_targets(('DA09','DB07','SMDVR','VA01'),np.eye(4),(d,np.zeros_like(d)))
    assert set(t.indices)=={0,1} and np.all(t.posterior_fraction>=.5)
    np.testing.assert_allclose(t.weights.sum(),1.)


def report(kind,head_scale=1.):
    t=np.arange(0,20,.002);s=np.sin(2*np.pi*t)
    v=np.full((len(t),3),-50.);v[:,2]+=head_scale*s
    q=np.zeros((len(t),11));q[:,:4]=.002*s[:,None]
    muscles=np.full((len(t),95),.5)
    tail=np.asarray(muscle_longitudinal_positions())>=.6
    dorsal=np.asarray([n.startswith('d') for n in BODY_WALL_MUSCLE_NAMES])
    if kind in ('active','drive-only'):muscles[:,tail&dorsal]+=.02*s[:,None]
    if kind=='common':muscles[:,tail]+=.02*s[:,None]
    if kind=='drive-only':q[:]=0.
    if kind=='neural':q[:]=0.;v[:,:2]+=20*s[:,None]
    body=SimpleNamespace(joint_angles=q,muscle_activation=q.copy())
    return posterior_motor_metrics(v,muscles,body,.002,.001,('DA09','DB07','SMDVR'),.75,
        targets=MotorTargets(np.array([0,1]),np.array([.6,.4]),np.ones(2)),
        projection=build_muscle_projection(11,head_at_end=True),release_threshold=np.full(3,-45.),release_slope=np.full(3,5.))


def test_passive_bends_common_mode_and_neural_only_cannot_qualify():
    assert report('active')['posterior_motor_qualified']
    assert not report('active')['body_amplitude_qualified']
    for kind in ('passive','common','neural','drive-only'):
        assert not report(kind)['posterior_motor_qualified']
    assert report('neural')['score']<=.05
    a,b=report('passive',1.),report('passive',40.)
    assert a['posterior_recruitment_score']==b['posterior_recruitment_score']==0.
    assert a['score']==b['score']
    active=report('active');assert posterior_motor_refinement_agrees(active,active)
    bad=dict(active,posterior_net_muscle_median_p2p=active['posterior_net_muscle_median_p2p']*1.4)
    assert not posterior_motor_refinement_agrees(active,bad)


def test_sources_use_repeated_internal_release_not_tonic_depolarization():
    t=np.arange(0,20,.002);v=np.full((len(t),3),-30.)
    v[:,2]=-45+20*np.sin(2*np.pi*t)
    assert np.array_equal(rhythmic_release_sources(v,np.full(3,-45.),np.full(3,5.)),[2])


class PathRng:
    def random(self):return .2
    def choice(self,a,size=None,replace=True,p=None):
        a=np.asarray(a)
        if size is not None:return a[:size]
        return a[-1] if p is None else a[0]
    def integers(self,n):return 0
    def uniform(self,a,b):return 1.5
    def normal(self,size=None):return 1. if size is None else np.ones(size)


def test_coordinated_proposal_uses_existing_directed_path_and_original_bounds():
    mask=np.ones((4,17),bool);mask[:,-1]=False
    edges=np.zeros((4,4),bool);edges[1,0]=True;edges[3,1]=True
    spec=JointFitSpec(mask,edges,np.zeros_like(edges))
    paths=chemical_driver_paths(edges,3,[0])
    assert paths==[((3,1),(1,0))]
    assert chemical_driver_paths(edges,0,[3])==[]
    targets=MotorTargets(np.array([3]),np.ones(1),np.ones(1))
    trial=posterior_motor_proposal(np.zeros(spec.size),spec,PathRng(),.3,16,targets=targets,driver_cells=[0])
    a=spec.counts['channel_densities']
    assert np.all(trial[a:]>0) and np.count_nonzero(trial[:a])==4
    assert np.all(trial[:3*16]==0)
    assert abs(trial).max()<=np.log(16)
