import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from r_elegans.wall_budget import WallDeadline,check_deadline,validation_reserve,can_search
from scripts.total_job_budget import run_group
from r_elegans.connection_amplitude import amplitude_score
from r_elegans.neural_timing import tail_score


def test_validation_is_reserved_and_overrun_lookahead_stops_search():
    reserve=validation_reserve(1800)
    assert reserve==4200
    assert can_search(8000,reserve,150)
    assert not can_search(4700,reserve,150)
    with pytest.raises(WallDeadline):check_deadline(time.monotonic()-1)
    check_deadline(None)


@pytest.mark.parametrize('parent_exits',[False,True])
def test_watchdog_cleans_up_descendants_even_after_leader_exit(tmp_path,parent_exits):
    ready=tmp_path/'ready';escaped=tmp_path/'escaped'
    child=f"from pathlib import Path; import time; Path({str(ready)!r}).write_text('ready'); time.sleep(1.2); Path({str(escaped)!r}).write_text('escaped')"
    code=f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep({.2 if parent_exits else 5})"
    started=time.monotonic()
    exit_code,timed_out=run_group([sys.executable,'-c',code],.5)
    assert time.monotonic()-started<2.5
    assert ready.exists()  # The descendant actually started; this isn't a vacuous kill test.
    assert timed_out is not parent_exits
    assert exit_code==(0 if parent_exits else 124)
    time.sleep(1.25)
    assert not escaped.exists()


def test_tail_objective_retains_frequency_and_middle_head():
    t=np.linspace(0,60,6001)
    def trace(frequency,tail=.00135,middle=.02):
        amplitudes=np.r_[np.full(4,tail),np.full(4,middle),np.full(3,.01)]
        return dict(dense_time_s=t,dense_joints=np.sin(2*np.pi*frequency*t)[:,None]*amplitudes,
                    dense_voltage_mV=5*np.sin(2*np.pi*frequency*t)[:,None])
    baseline=amplitude_score(trace(1.8),0,window_seconds=15)
    improved=tail_score(trace(1.8,tail=.0016),0,target_hz=1.8,window_seconds=15,baseline=baseline)
    assert improved['target_passed'] and improved['data_loss']==0
    fast=tail_score(trace(2.4,tail=.0016),0,target_hz=1.8,window_seconds=15,baseline=baseline)
    assert not fast['safety_passed']
    damaged=tail_score(trace(1.8,tail=.0016,middle=.01),0,target_hz=1.8,window_seconds=15,baseline=baseline)
    assert not damaged['middle_head_preserved'] and not damaged['safety_passed']


def test_control_search_uses_extra_budget_and_retains_full_validation():
    from r_elegans.wall_budget import control_search_deadline
    reserve=3600.
    short=control_search_deadline(4*3600.,reserve)
    long=control_search_deadline(10*3600.,reserve)
    assert long-short==6*3600.
    assert 10*3600.-long==reserve+600.
    with pytest.raises(ValueError):control_search_deadline(36000,0)
