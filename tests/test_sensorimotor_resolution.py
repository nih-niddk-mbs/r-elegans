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
