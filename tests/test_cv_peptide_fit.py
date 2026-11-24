"""E124 leakage/selection/parameter/budget and checkpoint regression checks."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
from scripts import run_cv_peptide_fit as fit
from scripts.cv_fit_core import group_directions,raw_peptide_gains,penalty,choose,admit_probe,full_window_checks,Evaluator
from scripts.stable_atlas_policy import Coordinates

REPO=Path(__file__).resolve().parents[1]


def sample_rows():
    rows=[]
    for i in range(110):
        subject=f'w{i:03d}';trials=[]
        if i<87:
            for j in range(7+(i<41)):
                trials.append(dict(subject=subject,target=f'T{(i*7+j)%20}',z={f'R{k}':float(i+j+k) for k in range(8)}))
        rows.append(dict(subject=subject,trials=trials))
    return rows


def test_five_folds_cover87_animals_once_and_heldout_changes_do_not_change_train():
    rows=sample_rows();folds,targets=fit.make_folds(rows)
    held=[s for f in folds for s in f['subjects']['heldout_subjects']]
    assert len(held)==len(set(held))==87 and len(targets)==20
    withheld=set(folds[0]['subjects']['heldout_subjects'])
    changed=sample_rows()
    for row in changed:
        if row['subject'] in withheld:
            for t in row['trials']:t['z']={k:100000. for k in t['z']}
    other,_=fit.make_folds(changed)
    np.testing.assert_array_equal(folds[0]['stats']['train']['mean_z'],other[0]['stats']['train']['mean_z'])
    assert folds[0]['panel']==other[0]['panel']
    assert not set(folds[0]['subjects']['train_subjects'])&withheld


def test_only_full_training_descent_selects_final_model():
    assert fit.training_choice(.8,1.,True)
    assert not fit.training_choice(1.1,1.,True)
    assert not fit.training_choice(.5,1.,False)
    current=dict(name='parent',safe=True,score=1.)
    assert choose(current,[dict(name='unsafe',safe=False,score=.1),dict(name='good',safe=True,score=.9)])['name']=='good'


def test_unknown_groups_cover_codec_without_changing_fixed_fields():
    a=fit.load_arrays(REPO/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz');codec=Coordinates(a)
    groups=group_directions(codec)
    np.testing.assert_array_equal(sum(groups.values()),np.ones(codec.size))
    changed=codec.apply(.01*groups['inward'])
    assert codec.audit(changed,.01*groups['inward'])['passed']
    for k in a:
        if not k.startswith('params_'):np.testing.assert_array_equal(a[k],changed[k])
    a,b=raw_peptide_gains(.05)
    assert np.logaddexp(0.,a)>np.logaddexp(0.,-9.)>np.logaddexp(0.,b)
    with pytest.raises(ValueError):raw_peptide_gains(.2)


def test_admission_preserves_final_validation_time():
    assert admit_probe(0.,1000.,3000.,300.,2400.)
    assert not admit_probe(0.,1000.,3000.,600.,2400.)
    assert not admit_probe(900.,1000.,5000.,300.,2400.)


def test_prospective_window_check_is_phase_robust_but_detects_amplitude_error():
    def waves(dt,phase=0.,gain=1.):
        t=(np.arange(round(60/dt))+1)*dt
        return -25.+gain*np.column_stack([10*np.sin(2*np.pi*t/.23317+phase),9*np.sin(2*np.pi*t/.21913+phase)])
    low=waves(.0005);fine=waves(.00025,.4)
    assert all(r['passed'] for r in full_window_checks(low,fine))
    assert not all(r['passed'] for r in full_window_checks(low,waves(.00025,.4,1.2)))


def test_timeout_still_writes_audited_incomplete_receipt(tmp_path,monkeypatch):
    from scripts import total_job_budget
    monkeypatch.setattr(fit,'BASE',tmp_path)
    monkeypatch.setattr(fit,'active_jobs',lambda:[])
    budgets=[]
    def timeout(command,seconds):budgets.append(seconds);return 124,True
    monkeypatch.setattr(total_job_budget,'run_group',timeout)
    root=tmp_path/'run';monkeypatch.setattr(sys,'argv',['fit','--output-dir',str(root)])
    assert fit.main()==2
    receipt=json.loads((root/'budget_receipt.json').read_text())
    assert receipt['timed_out'] and receipt['audit_complete'] and not receipt['all_steps_finished']
    assert 450*60<budgets[0]<=455*60


def test_status_and_report_do_not_start_work(tmp_path,monkeypatch):
    monkeypatch.setattr(fit,'BASE',tmp_path);monkeypatch.setattr(fit,'active_jobs',lambda:[])
    monkeypatch.setattr(fit.subprocess,'Popen',lambda *a,**k:pytest.fail('Read-only command launched'))
    monkeypatch.setattr(sys,'argv',['fit','--status']);fit.main()
    monkeypatch.setattr(sys,'argv',['fit','--output-dir',str(tmp_path),'--report']);fit.main()
    assert list(tmp_path.iterdir())==[]


def test_existing_job_refuses_launch_before_creating_output(tmp_path,monkeypatch):
    monkeypatch.setattr(fit,'BASE',tmp_path)
    monkeypatch.setattr(fit,'active_jobs',lambda:[dict(pid=123,output_dir='existing')])
    root=tmp_path/'new'
    monkeypatch.setattr(sys,'argv',['fit','--output-dir',str(root)])
    with pytest.raises(SystemExit):fit.main()
    assert not root.exists()


def test_shared_lock_refuses_second_launcher(tmp_path,monkeypatch):
    import fcntl
    monkeypatch.setattr(fit,'BASE',tmp_path);monkeypatch.setattr(fit,'active_jobs',lambda:[])
    root=tmp_path/'new';monkeypatch.setattr(sys,'argv',['fit','--output-dir',str(root)])
    with (tmp_path/'.cv-peptide-validation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(SystemExit):fit.main()
    assert not root.exists()


def test_complete_peptide_checkpoint_round_trip_with_fitted_balance(tmp_path,monkeypatch):
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import ConductanceNetworkState
    from r_elegans.brain.peptide_state_replay import FullState
    import r_elegans.brain.peptide_state_replay as module
    def tiny_engine(config,rmd,**options):
        def anchors(p,f,e,i,tau):
            n=302
            return FullState(ConductanceNetworkState(jnp.full((4,n),-35.),jnp.full((4,n,35),.5),jnp.full((4,n),.05),jnp.zeros((4,n))),jnp.full((4,n),.5)),dict(test_only=True)
        def replay(p,f,e,i,tau,start,gs):
            frames=12000;n=302;b=len(gs);t=np.arange(frames)*.001
            v=np.broadcast_to((-25+10*np.sin(2*np.pi*t/.23))[None,None,:,None],(4,b,frames,2)).copy()
            final=FullState(jax.tree.map(lambda a:jnp.broadcast_to(a[:,None],(4,b)+a.shape[1:]),start.neural),jnp.full((4,b,n),.5))
            means=np.broadcast_to(np.asarray(gs)[None],(4,b,n)).copy()
            return means,final,(v,np.zeros((4,b,frames)),np.zeros((4,b,frames)))
        return anchors,replay
    monkeypatch.setattr(module,'make_engine',tiny_engine)
    shutil.copy2(REPO/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz',tmp_path/'parent.npz')
    shutil.copy2(REPO/'r_elegans/assets/extrasynaptic_v1.npz',tmp_path/'peptide_asset.npz')
    evaluator=Evaluator(tmp_path,'peptide');x=np.zeros(evaluator.codec.size)
    data,row=evaluator.predict(x,.05,['RMDL'],0,tmp_path/'prediction')
    assert row['checks']['safe'] and data['response'].shape==(4,1,302)
    evaluator.save_complete_model(x,.05,row['path'],tmp_path/'complete.npz')
    from r_elegans.brain.peptide_checkpoint import load_model
    _,_,_,pp,exc,inh,states,values=load_model(tmp_path/'complete.npz')
    assert bool(values['extrasynaptic_enabled']) and np.asarray(exc).max()>0
    raw_exc,raw_inh=raw_peptide_gains(.05)
    np.testing.assert_array_equal(pp.raw_exc,np.full(len(pp.raw_exc),raw_exc))
    np.testing.assert_array_equal(pp.raw_inh,np.full(len(pp.raw_inh),raw_inh))
    np.testing.assert_array_equal(states.peptide,np.full((4,302),.5))
