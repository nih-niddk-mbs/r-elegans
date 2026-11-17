import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from r_elegans.relay_range import parameter_groups
from r_elegans.cma_curriculum import decode_groups


def test_recentered_group_stops_at_original_bound():
    ids=('AWAL','AIYL','RIAL');channels=np.ones((3,17),bool);channels[:,16]=False
    chemical=np.ones((3,3),bool);ci=np.nonzero(channels)
    size=int(channels.sum()+chemical.sum());origin=np.zeros(size)
    outward=(ci[0]==1)&np.isin(ci[1],[0,1,2,3,4,5,9,10,11,12,13,15])
    origin[np.flatnonzero(outward)]=.6
    lower=np.full(size,-np.log(2.));upper=-lower
    groups=parameter_groups(SimpleNamespace(channels=channels,chemical=chemical),ids,origin,lower,upper)
    group=next(g for g in groups if g['name']=='density:AIYL:outward')
    assert group['upper']<.4
    np.testing.assert_allclose(group['upper'],np.log(2.)-.6)
    decoded=decode_groups(np.ones(len(groups)),groups,origin,lower,upper)
    assert np.all(decoded<=upper+1e-12)
    np.testing.assert_allclose(decoded[group['indices']],np.log(2.))


@pytest.mark.parametrize('optimization_minutes',[None,480])
def test_continuation_budget_dispatch(tmp_path,monkeypatch,optimization_minutes):
    import scripts.total_job_budget as runner
    out=tmp_path/'out';calls=[]
    def run(cmd,seconds):
        calls.append(cmd)
        if 'scripts.audit_rial_sensory' in cmd:
            audit=tmp_path/'out-audit';audit.mkdir()
            (audit/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
        else:(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,validation_complete=True,sensory_relay_range_component_qualified=True)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','relay-range','--continue-relay','--total-minutes','660' if optimization_minutes else '180','--output-dir',str(out)]+(['--optimization-minutes','480'] if optimization_minutes else []))
    assert runner.main()==0 and '--continue-relay' in calls[0]
    if optimization_minutes:assert calls[0][calls[0].index('--optimization-minutes')+1]=='480.0'
    assert json.loads((out/'deadline.json').read_text())['experiment']==('E089' if optimization_minutes else 'E088')
    assert json.loads((out/'budget_receipt.json').read_text())['post_audit_component_qualified']


def test_continuation_rejects_other_objectives_before_writing_output(tmp_path,monkeypatch):
    import scripts.run_rial_sensory as runner
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['fit','--continue-relay','--objective','body','--output-dir',str(out),'--source-run',str(tmp_path/'source')])
    with pytest.raises(SystemExit) as e:runner.main()
    assert e.value.code==2 and not out.exists()
