import json
from pathlib import Path
import numpy as np
from scripts.report_rial_sensory import audit_details


def test_details_distinguish_selected_loss_and_physical_bound_room(tmp_path):
    run=tmp_path/'run';run.mkdir();selected=run/'candidate';selected.mkdir()
    audit=tmp_path/'run-audit';audit.mkdir();(run/'optimizer/final').mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value))
    feature=dict(AIYL_release_q01=.97,AIYL_release_q99=.981)
    paired={k:{category:[feature] for category in ('active','response','late')} for k in ('no_cue','cue')}
    metrics=dict(loss=1.2,relay_response_mean_changes=[-.000021],relay_operating_range_passed=False)
    row=dict(output=str(selected),paired_features=paired,metrics=metrics)
    summary=dict(experiment='fixture',final_target=.02,baseline=row,selected=row)
    write(selected/'summary.json',dict(metrics=metrics))
    write(run/'generation_history.json',[dict(generation=3,target=.02,minimum_loss=.9,selected_evaluation=str(selected))])
    write(run/'parameter_groups.json',[dict(name='edge',indices=[0],lower=-.4,upper=.4)])
    write(audit/'parameters.json',dict(groups=[dict(name='edge',near_group_bound=True)]))
    write(run/'optimizer/final/metadata.json',dict(stage=0,generation=4,pending=[[0.]]))
    write(run/'partial_generation.json',dict(evaluations=[dict(status='time_budget')]))
    np.savez(run/'parameter_origin.npz',lower=[-.7],upper=[.7])
    np.savez(selected/'coordinates.npz',offsets=[.3])
    before={p:p.read_bytes() for p in run.rglob('*') if p.is_file()}
    report=audit_details(run,summary)
    np.testing.assert_allclose(report['operating_range']['selected']['upper_release_headroom'],.019)
    assert report['recent_generations'][0]['selected_loss']==1.2
    assert report['recent_generations'][0]['population_minimum']==.9
    np.testing.assert_allclose(report['near_bound_groups'][0]['minimum_original_lower_room'],1.)
    np.testing.assert_allclose(report['near_bound_groups'][0]['minimum_original_upper_room'],.4)
    assert report['optimizer']['pending_points']==1
    assert report['partial_generation_statuses']=={'time_budget':1}
    assert all(p.read_bytes()==v for p,v in before.items())
