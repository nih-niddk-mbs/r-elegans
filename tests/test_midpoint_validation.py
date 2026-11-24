import json
import sys
import numpy as np
import pytest
from scripts import run_midpoint_validation as runner


def test_cycle_dispersion_detects_insufficient_phase_sampling():
    a=np.ones((16,2,3))*.01
    assert runner.cycle_sampling(a)['passed']
    a[:4]*=5
    assert not runner.cycle_sampling(a)['passed']


def test_numerical_benchmark_selects_coarsest_safe_improvement_only():
    cube=np.ones((4,3,302))*.001
    data={name:dict(response=cube*factor) for name,factor in [('legacy-reference',1),('legacy-own',2),('legacy-coarse',2),('midpoint-0',1.01),('midpoint-1',1.001),('midpoint-2',1.0001)]}
    checks={name:dict(checks=dict(finite=True,duplicate=True)) for name in data}
    level,rows=runner.choose_clock(data,checks)
    assert level==0 and all(r['passed'] for r in rows)
    checks['midpoint-0']['checks']['finite']=False
    assert runner.choose_clock(data,checks)[0]==1
    data['midpoint-1']['response']=cube*5
    assert runner.choose_clock(data,checks)[0] is None


def test_readonly_refuses_launch_and_creates_no_files(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'jobs',lambda:[])
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:pytest.fail('Read-only launched'))
    for args in (['--status'],['--output-dir',str(tmp_path),'--report']):
        monkeypatch.setattr(sys,'argv',['probe']+args);runner.main()
    assert not list(tmp_path.iterdir())


def test_timeout_still_audits(tmp_path,monkeypatch):
    from scripts import total_job_budget
    monkeypatch.setattr(runner,'jobs',lambda:[]);monkeypatch.setattr(runner,'BASE',tmp_path)
    monkeypatch.setattr(total_job_budget,'run_group',lambda *a:(124,True))
    monkeypatch.setattr(sys,'argv',['probe','--output-dir',str(tmp_path/'out')])
    assert runner.main()==2
    receipt=json.loads((tmp_path/'out/budget_receipt.json').read_text())
    assert receipt['timed_out'] and receipt['audit_complete'] and not receipt['all_steps_finished']


def test_existing_work_prevents_output_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'jobs',lambda:[dict(pid=123)])
    monkeypatch.setattr(sys,'argv',['probe','--output-dir',str(tmp_path/'out')])
    with pytest.raises(SystemExit):runner.main()
    assert not (tmp_path/'out').exists()


def test_shared_lock_blocks_duplicate_before_output_creation(tmp_path,monkeypatch):
    import fcntl
    monkeypatch.setattr(runner,'jobs',lambda:[]);monkeypatch.setattr(runner,'BASE',tmp_path)
    root=tmp_path/'new';monkeypatch.setattr(sys,'argv',['probe','--output-dir',str(root)])
    with (tmp_path/'.cv-peptide-validation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(SystemExit):runner.main()
    assert not root.exists()
