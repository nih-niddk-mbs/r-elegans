import json
import sys
import numpy as np
from r_elegans.nmj_source import targets_output,release_hold,protocol_settings,refinement_target,split_controls


def test_split_membership_and_reversible_holds():
    import jax.numpy as jnp
    import jax
    ids=('RMDL','RMDR','RMDVL','SMDDL','SMDVR','SMBDL','SMBVR','DB01','ASHL')
    voltage=jnp.full(len(ids),-30.);threshold=jnp.full(len(ids),-40.);slope=jnp.full(len(ids),5.)
    mean=jnp.linspace(.1,.3,len(ids));live=jax.nn.sigmoid((voltage-threshold)/slope)
    subsets=[]
    for group in ('RMD','SMD','SMB'):
        mask=np.asarray([targets_output(n,group) for n in ids]);subsets.append(mask)
        hold=release_hold(ids,threshold,slope,mean,group)
        expected=np.asarray(live).copy();expected[mask]=np.asarray(mean)[mask]
        np.testing.assert_allclose(hold(voltage,35.),expected)
        for t in (0.,29.9,40.,59.):np.testing.assert_array_equal(hold(voltage,t),live)
    np.testing.assert_array_equal(np.sum(subsets,axis=0),[1]*7+[0,0])
    assert not any(targets_output(n,'sham') for n in ids)


def test_head_split_refines_individual_group_and_excludes_controls():
    exp,cases,eligible=protocol_settings('head-split')
    assert exp=='E081' and len(cases)==6 and eligible==('RMD','SMD','SMB')
    def row(c,r,safe=True):return dict(case=c,metrics=dict(pre_cue_match=True,safety_passed=safe,during_to_sham_raw_range_ratio=r))
    rows=[row('all',0.),row('head',.1),row('RMD',.2),row('SMD',.8),row('SMB',1.)]
    assert refinement_target(rows,'head-split')=='RMD'
    rows[2]=row('RMD',.2,False)
    assert refinement_target(rows,'head-split')=='SMD'
    assert refinement_target([row('all',0.),row('head',.1)],'head-split') is None
    assert refinement_target(rows,'population')=='head'


def test_head_split_watchdog_dispatch_records_correct_protocol(tmp_path,monkeypatch):
    import scripts.total_job_budget as runner
    out=tmp_path/'output';seen=[]
    def run(command,seconds):
        seen.append(command)
        (out/'summary.json').write_text(json.dumps(dict(execution_valid=True,validation_complete=True)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','head-split','--output-dir',str(out),'--source-run',str(tmp_path/'source'),'--total-minutes','45'])
    assert runner.main()==0
    assert seen[0][3]=='scripts.run_nmj_source'
    assert seen[0][-4:-2]==['--protocol','head-split']
    assert json.loads((out/'deadline.json').read_text())['experiment']=='E081'
    receipt=json.loads((out/'budget_receipt.json').read_text())
    assert receipt['total_minutes']==45 and receipt['all_steps_finished']


def test_split_requires_reproduced_combined_and_static_output_controls():
    def row(c,r):return dict(case=c,metrics=dict(pre_cue_match=True,safety_passed=True,recovery_amplitude_preserved=True,during_to_sham_raw_range_ratio=r))
    assert split_controls([row('head',.114),row('all',0.)])['passed']
    assert not split_controls([row('RMD',.1),row('head',1.),row('all',0.)])['passed']
    assert not split_controls([row('head',.114)])['passed']
