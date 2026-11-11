import numpy as np
import pytest
from scripts.assess_sensorimotor_resolution import cue_effect_difference


def test_resolution_compares_cue_effect_not_sham_posture_and_aligns_time_grids():
    def make(dt,posture):
        t=np.arange(0,1+dt/2,dt)
        sham=np.full((len(t),11),posture)
        active=sham+.0001*t[:,None]
        return {'active':dict(dense_time_s=t,dense_joints=active),
                'sham':dict(dense_time_s=t,dense_joints=sham)}
    coarse=make(.1,.1);fine=make(.05,.2)
    assert cue_effect_difference(coarse,fine)['maximum_cue_effect_difference_rad']<1e-15
    fine['active']['dense_joints'][:,3]+=.00002
    result=cue_effect_difference(coarse,fine)
    assert result['maximum_cue_effect_difference_rad']==pytest.approx(.00002)
    assert result['per_joint_difference_rad'][3]==pytest.approx(.00002)
    fine['active']['dense_time_s']+=.01
    with pytest.raises(ValueError,match='cover'):cue_effect_difference(coarse,fine)


def test_split_cases_vary_one_step_family_and_compare_within_branch():
    from scripts.assess_sensorimotor_resolution import integration_cases
    cases=integration_cases('split',.002,.1)
    lookup={name:(body,brain,reference) for name,body,brain,reference in cases}
    assert lookup['brain-half']==(.002,.05,'coarse')
    assert lookup['brain-quarter']==(.002,.025,'brain-half')
    assert lookup['body-half']==(.001,.1,'coarse')
    assert lookup['body-quarter']==(.0005,.1,'body-half')
    assert integration_cases('matched',.002,.1)==[
        ('coarse',.002,.1,None),('half',.001,.05,'coarse'),('quarter',.0005,.025,'half')]
    with pytest.raises(ValueError):integration_cases('unknown',.002,.1)
