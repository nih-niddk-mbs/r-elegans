import json
import numpy as np
import pytest
from scripts.experiment_records import file_sha256
from scripts.sensorimotor_resume import read_completed,MATCH_KEYS


@pytest.fixture
def saved(tmp_path):
    cfg={k:None for k in MATCH_KEYS}
    cfg.update(stage='contract_relax',maximum_factor=2.,base_sensory_config={
        'chemical_gain_nS':.005,'b_bend_gain_nS':.005,'dva_gain_nS':.002},
        arguments=dict(sigma=.03,bend_target_rad=.00003,dt=.002,brain_step_ms=.1))
    x=np.array([.1,-.2]);s=np.log([1.2,.9,1.1]);g=np.exp(s).tolist()
    (tmp_path/'network_best.npz').write_bytes(b'checkpoint fixture')
    result=dict(total=.7)
    summary=dict(experiment='E055',status='completed',iterations=3,selected_iteration=2,
        checkpoint_sha256=file_sha256(tmp_path/'network_best.npz'),best=result,sensory_gain_multipliers=g)
    sensory=dict(base=cfg['base_sensory_config'],gain_multipliers=g)
    for key,i in (('chemical_gain_nS',0),('b_bend_gain_nS',1),('dva_gain_nS',2)):
        sensory[key]=float(sensory['base'][key]*g[i])
    rng=np.random.default_rng(5500);rng.normal(size=12)
    data={'configuration.json':cfg,'summary.json':summary,'sensory_parameters.json':sensory,
        'best.json':dict(iteration=2,result=result,sensory_gain_multipliers=g),
        'rng_state.json':rng.bit_generator.state}
    for name,value in data.items():(tmp_path/name).write_text(json.dumps(value))
    np.savez(tmp_path/'search_state.npz',offsets=x,sensory_offsets=s,iteration=3)
    np.savez(tmp_path/'best_state.npz',offsets=x,sensory_offsets=s,iteration=2)
    return tmp_path,cfg,x,s,rng


def test_resume_restores_both_parameter_families_and_post_rejection_rng(saved):
    root,cfg,x,s,rng=saved
    a,b,i,resumed,_=read_completed(root,cfg,2)
    np.testing.assert_array_equal(a,x);np.testing.assert_array_equal(b,s)
    assert i==3  # Rejected proposal after selected_iteration2 must still count.
    np.testing.assert_array_equal(resumed.normal(size=32),rng.normal(size=32))


def test_changed_objective_or_checkpoint_rejected(saved):
    root,cfg,*_=saved
    changed=json.loads(json.dumps(cfg));changed['arguments']['bend_target_rad']=.003
    with pytest.raises(ValueError,match='objective'):read_completed(root,changed,2)
    (root/'network_best.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):read_completed(root,cfg,2)


def test_gains_cannot_be_reset_or_drift_outside_original_bounds(saved):
    root,cfg,x,s,_=saved
    np.savez(root/'search_state.npz',offsets=x,sensory_offsets=np.zeros(3),iteration=3)
    with pytest.raises(ValueError,match='states disagree'):read_completed(root,cfg,2)
    np.savez(root/'search_state.npz',offsets=x,sensory_offsets=[np.log(3),0.,0.],iteration=3)
    with pytest.raises(ValueError,match='bounds'):read_completed(root,cfg,2)


def test_inconsistent_saved_effective_sensory_gain_rejected(saved):
    root,cfg,*_=saved
    path=root/'sensory_parameters.json';data=json.loads(path.read_text())
    data['chemical_gain_nS']=.5;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='effective sensory gain'):read_completed(root,cfg,2)


def test_changed_training_cues_require_fresh_stage_not_rng_resume(saved):
    root,cfg,*_=saved
    changed=json.loads(json.dumps(cfg));changed['arguments']['training_cues']='nominal-low-dose'
    with pytest.raises(ValueError,match='fresh-stage warm start'):read_completed(root,changed,2)
    # Older nominal configurations remain compatible with the explicit new default.
    changed['arguments']['training_cues']='nominal'
    assert read_completed(root,changed,2)[2]==3


def test_precision_or_margin_change_requires_warm_start(saved):
    root,cfg,*_=saved
    for key,value in [('precision','float64'),('training_margin',1.02)]:
        changed=json.loads(json.dumps(cfg));changed['arguments'][key]=value
        with pytest.raises(ValueError,match=key):read_completed(root,changed,2)


def test_operational_reference_hash_is_required_for_float64_resume(saved):
    root,cfg,*_=saved
    cfg['arguments'].update(precision='float64',training_margin=1.02)
    cfg['parameterization']='operational-parent-v1'
    (root/'parameter_reference.npz').write_bytes(b'parameter reference fixture')
    np.savez(root/'parameter_origin.npz',offsets=np.zeros(2))
    cfg['parameter_reference_sha256']=file_sha256(root/'parameter_reference.npz')
    cfg['parameter_origin_sha256']=file_sha256(root/'parameter_origin.npz')
    (root/'configuration.json').write_text(json.dumps(cfg))
    assert read_completed(root,cfg,2)[2]==3
    (root/'parameter_reference.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='reference checksum'):read_completed(root,cfg,2)
