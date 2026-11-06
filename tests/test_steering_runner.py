"""Signed steering jobs remain independent and start from matching forward seeds."""
import json
from types import SimpleNamespace

import pytest

from scripts import run_motor_curriculum as runner


@pytest.mark.parametrize('seed,sign',[(0,1),(1,-1)])
def test_standalone_steering_job_uses_forward_checkpoint_and_expected_bias(tmp_path,monkeypatch,seed,sign):
    source=tmp_path/'wave-source';checkpoint=source/f'seed-{seed}/wave/actor_best.npz'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'forward actor')
    (source/'summary.json').write_text(json.dumps(dict(workers=[dict(seed=seed,stages=dict(
        wave=dict(passed=True,status='completed',exit_code=0)))])))
    launched=[]
    class Process:
        pid=123;returncode=0
        def __init__(self,command,**kwargs):
            launched.append(command);folder=runner.Path(command[command.index('--output-dir')+1]);folder.mkdir()
            (folder/'summary.json').write_text(json.dumps(dict(status='completed',error=None,passed=True)))
        def poll(self):return self.returncode
        def wait(self,timeout=None):return self.returncode
    monkeypatch.setattr(runner.sys,'platform','linux')
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda _:{0},raising=False)
    monkeypatch.setattr(runner,'physical_core_cpus',lambda _:([0],True))
    monkeypatch.setattr(runner,'capture_run_record',lambda *a:None)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0))
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner.signal,'signal',lambda *a:None)
    monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    monkeypatch.setattr(runner.sys,'argv',['runner','--steering','--minutes','480','--workers','1',
        '--cores-per-worker','1','--seed-start',str(seed),'--resume-run',str(source),'--output-dir',str(tmp_path/'results')])
    assert runner.main()==0
    assert len(launched)==1
    command=launched[0]
    assert command[2].endswith('train_steering.py')
    assert command[command.index('--initial-actor')+1]==str(checkpoint)
    assert float(command[command.index('--bias')+1])==pytest.approx(sign*.006)
