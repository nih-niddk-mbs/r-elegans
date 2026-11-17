import argparse
import json
import sys
import time
import pytest
from scripts.relay_parent import add_arguments,validate


@pytest.mark.parametrize('extra',[
    ['--parent-sha256','a'*64],
    ['--parent-sha256','wrong','--experiment-id','E090','--seed','9000'],
    ['--parent-sha256','a'*64,'--experiment-id','wrong','--seed','9000'],
    ['--parent-sha256','a'*64,'--experiment-id','E090','--seed','-1'],
])
def test_incomplete_or_invalid_pin_is_rejected(extra):
    p=argparse.ArgumentParser();add_arguments(p)
    with pytest.raises(SystemExit) as e:validate(p,p.parse_args(extra),True)
    assert e.value.code==2


def test_explicit_pin_requires_relay_scope_and_keeps_values():
    p=argparse.ArgumentParser();add_arguments(p)
    args=p.parse_args(['--parent-sha256','a'*64,'--experiment-id','E090','--seed','9000'])
    assert validate(p,args,True)
    assert args.parent_sha256=='a'*64 and args.seed==9000
    with pytest.raises(SystemExit):validate(p,args,False)


def test_parent_pin_is_forwarded_and_recorded_in_watchdog(tmp_path,monkeypatch):
    import scripts.total_job_budget as runner
    out=tmp_path/'out';calls=[]
    def run(cmd,seconds):
        calls.append(cmd)
        if 'scripts.audit_rial_sensory' in cmd:
            folder=tmp_path/'out-audit';folder.mkdir()
            (folder/'summary.json').write_text(json.dumps(dict(audit_complete=True,integrity_passed=True)))
        else:(out/'summary.json').write_text(json.dumps(dict(execution_valid=True,validation_complete=True)))
        return 0,False
    monkeypatch.setattr(runner,'run_group',run)
    monkeypatch.setattr(sys,'argv',['budget','--pipeline','relay-range','--continue-relay','--optimization-minutes','480',
        '--total-minutes','660','--output-dir',str(out),'--parent-sha256','a'*64,'--experiment-id','E090','--seed','9000'])
    assert runner.main()==0
    assert calls[0][calls[0].index('--parent-sha256')+1]=='a'*64
    record=json.loads((out/'deadline.json').read_text())
    assert record['experiment']=='E090' and record['seed']==9000 and record['parent_sha256']=='a'*64


@pytest.mark.parametrize('case',['missing_receipt','audit_incomplete','integrity_failed','wrong_hash'])
def test_unusable_source_is_rejected_before_inputs_are_copied(tmp_path,monkeypatch,case):
    import scripts.run_rial_sensory as runner
    source=tmp_path/'source';source.mkdir();out=tmp_path/'output';out.mkdir()
    summary=dict(experiment='E089',objective='relay-range',execution_valid=True,validation_complete=True,
                 selected=dict(checkpoint_sha256='b'*64 if case=='wrong_hash' else 'a'*64))
    (source/'summary.json').write_text(json.dumps(summary))
    if case!='missing_receipt':
        (source/'budget_receipt.json').write_text(json.dumps(dict(all_steps_finished=True,
            audit_complete=case!='audit_incomplete',audit_integrity_passed=case!='integrity_failed')))
    (out/'deadline.json').write_text(json.dumps(dict(coordinator_finish_by=time.monotonic()+40000,total_minutes=660)))
    monkeypatch.setattr(runner,'capture_run_record',lambda *args,**kwargs:None)
    monkeypatch.setattr(sys,'argv',['fit','--objective','relay-range','--continue-relay','--optimization-minutes','480',
        '--source-run',str(source),'--output-dir',str(out),'--parent-sha256','a'*64,'--experiment-id','E090','--seed','9000'])
    with pytest.raises((ValueError,FileNotFoundError)):runner.main()
    assert not (out/'inputs').exists()
    assert json.loads((source/'summary.json').read_text())==summary
