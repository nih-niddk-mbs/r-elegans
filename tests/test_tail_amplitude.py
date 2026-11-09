from types import SimpleNamespace
import numpy as np
from r_elegans.brain.tail_amplitude import tail_amplitude_metrics,tail_refinement_agrees,tail_proposal,tail_focus_cells
from r_elegans.brain.joint_autonomous_fit import JointFitSpec


def metric(tail=True,head=False,voltage_scale=1.):
    t=np.arange(0,20,.002);wave=np.sin(2*np.pi*t)
    q=np.zeros((len(t),11))
    if tail:q[:,:4]=.002*wave[:,None]
    if head:q[:,8:]=.02*wave[:,None]
    v=-50+voltage_scale*wave[:,None]*np.ones((1,3))
    body=SimpleNamespace(joint_angles=q,muscle_activation=q.copy())
    return tail_amplitude_metrics(v,np.full((len(t),95),.5),body,.002,.001,('A','B','C'),.75,focus_cells=[1,2])


def test_tail_can_qualify_without_head_middle_or_body_wide_qualification():
    m=metric()
    assert m['tail_amplitude_qualified'] and not m['body_amplitude_qualified']
    assert not m['locomotion_demonstrated'] and not m['wave_qualified']
    assert tail_refinement_agrees(m,m)
    bad=dict(m,regional_joint_median_p2p_rad=[.008,0,0])
    assert not tail_refinement_agrees(m,bad)


def test_head_motion_or_large_neural_rhythms_cannot_qualify_tail():
    a,b=metric(False,True,.1),metric(False,True,20)
    assert not a['tail_amplitude_qualified'] and not b['tail_amplitude_qualified']
    assert 0<=a['score']<b['score']<=.05
    assert b['tail_coverage_score']==0


def test_tail_focus_uses_posterior_projection_and_sparse_bounded_proposals():
    d=np.eye(11);v=np.zeros_like(d)
    weights=np.eye(11)
    assert np.array_equal(tail_focus_cells(weights,(d,v)),[0,1,2,3])
    channels=np.ones((20,3),bool);chemical=np.ones((20,20),bool)
    gap=~np.eye(20,dtype=bool);spec=JointFitSpec(channels,chemical,gap)
    best=np.zeros(spec.size);rng=np.random.default_rng(2000);sparse=0;seen=np.zeros(spec.size,bool)
    for _ in range(150):
        trial=tail_proposal(best,spec,rng,.2,16,focus_cells=[17,18,19])
        changed=trial!=best;seen|=changed;sparse+=changed.sum()<=8
        assert np.isfinite(trial).all() and abs(trial).max()<=np.log(16)
    assert sparse>120
    a,b,c=spec.counts.values()
    assert all(seen[start:end].any() for start,end in ((0,a),(a,a+b),(a+b,a+b+c)))
