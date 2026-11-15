import json
from scripts.report_cma_curriculum import candidate_diagnostics


def test_diagnostics_use_metadata_without_pruned_trajectories(tmp_path):
    for i,loss,uncued,ratio in [(0,.7,True,.999),(1,.2,False,.96)]:
        folder=tmp_path/'evaluations'/f'g-00000-c-{i:02d}';folder.mkdir(parents=True)
        m=dict(loss=loss,data_loss=loss,constraint_loss=0.,safe=True,uncued_movement_preserved=uncued,
            pre_movement_preserved=True,bend_amplitude_preserved=True,partial_cadence_passed=False,
            cadence_ratios=[dict(condition='stop_only',window='off',frequency_ratios=[ratio]*3)])
        (folder/'summary.json').write_text(json.dumps(dict(status='completed',metrics=m,checkpoint_sha256=str(i)*64)))
    result=candidate_diagnostics(tmp_path)
    assert result['candidates']==2 and result['eligible']==1
    assert result['best_unconstrained']['name']=='g-00000-c-01'
    assert result['best_eligible']['name']=='g-00000-c-00'
    assert result['failed_checks']['uncued']==1
    assert result['cadence_ratio_ranges']['stop_only/off']==dict(minimum=.999,maximum=.999)
