"""E069: prepare frozen saved-command capsule and run independent body replays."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from scripts.experiment_records import capture_run_record,file_sha256

PARENT='f024bd14b75e07a6c9c48bcc2bc03aed4b8354510c1d46ee8235e7575638f56e'
ACTOR='82f418ac0a0b078960c8bacd0e7a6bc7bddb99ddf68820dd6d8ac6bb8247dcb0'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--software-smoke',action='store_true')
    p.add_argument('--no-affinity',action='store_true')
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];source=a.source_run.resolve();out=a.output_dir.resolve()
    for root in (repo,source):
        if out==root or root in out.parents or out in root.parents:p.error('Separate external output required')
    metadata=json.loads((source/'inputs/manifest.json').read_text())
    source_summary=json.loads((source/'summary.json').read_text())
    if not source_summary['execution_valid'] or metadata['protocol']!='hybrid':raise ValueError('Completed hybrid source required')
    if not a.software_smoke and metadata['candidate_sha256']!=PARENT:raise ValueError('Pinned server hybrid required')
    if file_sha256(source/'inputs/candidate.npz')!=metadata['candidate_sha256'] or file_sha256(source/'inputs/actor.npz')!=ACTOR:
        raise ValueError('Candidate/readout hash mismatch')
    cfg=source/'hybrid-sham/configuration.json';trace=source/'hybrid-sham/trajectory.npz'
    body_files=['r_elegans/body/calibration.py','r_elegans/body/actuation.py','r_elegans/body/wet_surface.py','r_elegans/body/mechanics.py']
    records=list((source/'hybrid-sham/run_records').glob('*/source_record.json'))
    matches=[]
    for record in records:
        hashes=json.loads(record.read_text())['source_sha256']
        if all(hashes.get(name)==file_sha256(repo/name) for name in body_files):matches.append(record)
    if not matches:raise ValueError('No source invocation capsule matches the current body implementation')
    inputs_to_hash=[trace,cfg,source/'summary.json',source/'inputs/manifest.json',
                    source/'inputs/candidate.npz',source/'inputs/actor.npz',*matches,*[repo/n for n in body_files]]
    before={str(path):file_sha256(path) for path in inputs_to_hash}
    with np.load(trace,allow_pickle=False) as z:
        t=z['dense_time_s'];duration=.4 if a.software_smoke else 15.
        if not a.software_smoke and abs(float(t[-1])-60)>1e-9:raise ValueError('Sixty-second source required')
        if not np.allclose(np.diff(t),.002,rtol=0,atol=1e-10):raise ValueError('Uniform .002s source required')
        first=round((float(t[-1])-duration)/.002)
        segment=dict(source_time_s=t[first:],command=z['dense_command'][first:],joints=z['dense_joints'][first:],
                     drive=z['dense_filtered_drive'][first:],position=z['dense_position'][first:],heading=z['dense_heading'][first:])
    out.mkdir(parents=True,exist_ok=False);capsule=out/'inputs';capsule.mkdir();(out/'logs').mkdir()
    capture_run_record(out,repo,a)
    np.savez_compressed(capsule/'segment.npz',**segment)
    # Reproduce the source's legacy32 body template, then widen each scalar exactly.
    import jax
    jax.config.update('jax_enable_x64',False)
    from r_elegans.body import physical_muscle_body_params
    body=physical_muscle_body_params(12,medium='agar')
    document={key:float(value) for key,value in body._asdict().items() if key!='mechanics'}
    document['mechanics']={key:float(value) for key,value in body.mechanics._asdict().items()}
    (capsule/'body.json').write_text(json.dumps(document,indent=2)+'\n')
    for name,path in (('network.npz',source/'inputs/candidate.npz'),('actor.npz',source/'inputs/actor.npz'),('source_configuration.json',cfg)):
        shutil.copy2(path,capsule/name)
        if file_sha256(path)!=file_sha256(capsule/name):raise RuntimeError('Copy mismatch')
    manifest=dict(experiment='E069',source_run=str(source),source_hashes=before,
                  parent_sha256=metadata['candidate_sha256'],actor_sha256=ACTOR,software_smoke=a.software_smoke,
                  source_segment_s=[float(segment['source_time_s'][0]),float(segment['source_time_s'][-1])],
                  sha256={path.name:file_sha256(path) for path in capsule.iterdir()})
    (capsule/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if a.no_affinity:cpus=[None]*4
    else:
        from scripts.benchmark_cpu import physical_core_cpus
        available,known=physical_core_cpus(sorted(os.sched_getaffinity(0)))
        if not known or len(available)<4 or not shutil.which('taskset'):p.error('Four verified physical CPUs required')
        cpus=available[:4]
    cases=[('normal',1.,.002,False),('half-speed',.5,.002,False),('quarter-speed',.25,.002,False),
           ('eighth-speed',.125,.002,False),('normal-refined',1.,.001,False),
           ('eighth-refined',.125,.001,False),('held-command',1.,.002,True)]
    if a.software_smoke:cases=[cases[0],cases[1],cases[4]]
    running={};finished=[];started=time.monotonic();heartbeat=started
    def start(case,slot):
        name,speed,dt,held=case
        command=[sys.executable,'-u','-m','scripts.body_timing_worker','--inputs',str(capsule),
                 '--output-dir',str(out/name),'--speed',str(speed),'--dt',str(dt)]
        if held:command.append('--held')
        if cpus[slot] is not None:command=['taskset','-c',str(cpus[slot]),*command]
        log=(out/'logs'/(name+'.log')).open('w')
        proc=subprocess.Popen(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        running[slot]=(proc,log,case);print(json.dumps(dict(starting=name,pid=proc.pid,cpu=cpus[slot])),flush=True)
    def finish(slot):
        proc,log,case=running.pop(slot);log.close();name=case[0]
        if proc.returncode:raise RuntimeError('Replay failed; see '+str(out/'logs'/(name+'.log')))
        result=json.loads((out/name/'summary.json').read_text());finished.append(dict(case=name,result=result))
        print(json.dumps(dict(finished=name,cycles=result['metrics']['minimum_regional_joint_cycles'],source_match=result['source_match'])),flush=True)
    try:
        start(cases.pop(0),0)
        running[0][0].wait();finish(0)  # Source reproduction gates the timing interpretation.
        while cases or running:
            for slot in range(4):
                if slot in running and running[slot][0].poll() is not None:finish(slot)
                if slot not in running and cases:start(cases.pop(0),slot)
            if time.monotonic()-heartbeat>=30:
                print(json.dumps(dict(elapsed_minutes=(time.monotonic()-started)/60,running=[row[2][0] for row in running.values()])),flush=True);heartbeat=time.monotonic()
            if running:time.sleep(1)
    finally:
        for proc,log,_ in running.values():
            if proc.poll() is None:proc.terminate()
            proc.wait();log.close()
    if {str(path):file_sha256(path) for path in inputs_to_hash}!=before:raise RuntimeError('Source inputs changed during replay')
    rows={r['case']:r['result']['metrics'] for r in finished};checks=[]
    for normal,refined in [('normal','normal-refined'),('eighth-speed','eighth-refined')]:
        if refined not in rows:continue
        from r_elegans.body_timing import refinement_check
        with np.load(out/normal/'trajectory.npz') as x,np.load(out/refined/'trajectory.npz') as y:
            checks.append(dict(pair=[normal,refined],**refinement_check(x,y)))
    result=dict(experiment='E069',execution_valid=True,software_smoke=a.software_smoke,
                source_inputs_unchanged=True,finished=finished,refinement=checks,elapsed_seconds=time.monotonic()-started,
                interpretation='Diagnostic only: unchanged brain was not simulated; retimed replay cannot qualify autonomous movement. Inferred environmental torque is not an independent force-law validation.')
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for row in finished:
        metrics=row['result']['metrics']
        print(json.dumps(dict(case=row['case'],speed=row['result']['speed'],dt=row['result']['dt'],
            simulated_seconds=metrics['duration_s'],cycles_tail_middle_head=metrics['minimum_regional_joint_cycles'],
            last_window_filtered_cycles=metrics['windows'][-1]['regional_cycles']['drive'],
            maximum_force_residual=metrics['maximum_force_residual'],
            mean_below_drag_regularization_fraction=metrics['mean_below_drag_regularization_fraction'])),flush=True)
    print(json.dumps(dict(execution_valid=True,refinement=checks,report=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()
