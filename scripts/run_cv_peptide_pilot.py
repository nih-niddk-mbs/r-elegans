"""E122 one-fold, two-GPU integration pilot; not completed fivefold CV.

Train-only, bounded coordinate search. Held-out animals never select updates.
Maximum one hour includes setup, numerical checks, audit and process cleanup.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from scripts.experiment_records import capture_run_record, file_sha256
from scripts.stable_atlas_worker import atomic_json

REPO = Path(__file__).resolve().parents[1]
SEED = 1220
REFERENCE_SHA = '47a369a17f8e8c1387cd20f66c00f48db34b935591e043eb216731880c22482d'
UNAVAILABLE = ['Five independent fold fits and all-animal calibration',
    'Full 135-target atlas evaluation and independent full-loss replication',
    'Long-duration neural dynamics, body, swimming and sensory start/stop qualification',
    'Spatial diffusion, monoamines and measured peptide release kinetics',
    'Exact interrupted JIT/task resume and tested recovery after machine failure']
UNAVAILABLE += ['Three-level convergence of peptide coupling and individual-cell voltage histories; pilot has two grids and RMD traces only',
    'Peptide gain/time-constant optimization; both remain fixed priors in this pilot',
    'Batch-size/concurrency sweep on this new solver; two GPU arms are measured, scaling is not assumed']


def write(path,value):
    def convert(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    atomic_json(path,json.loads(json.dumps(value,default=convert,allow_nan=False)))


def physical_cpus():
    """Choose available distinct physical cores, respecting scheduler affinity."""
    allowed=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1))
    selected={}
    for cpu in allowed:
        top=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=tuple((top/f).read_text().strip() for f in ('physical_package_id','core_id')) if top.exists() else ('unknown',str(cpu))
        selected.setdefault(key,cpu)
    return list(selected.values())


def read_trials(path):
    import h5py
    from r_elegans.assets import load_asset_document
    from r_elegans.data.randi_atlas import load_randi_animal
    from r_elegans.atlas_evaluation import animal_trials
    path = Path(path); before = file_sha256(path)
    with h5py.File(path, 'r') as f:
        def value(key):
            v = f['general/subject/'+key][()]
            return (v.decode() if isinstance(v, bytes) else str(v)).strip()
        subject, genotype = value('subject_id'), value('genotype')
    if not subject or subject.lower() in ('unknown', 'none', 'nan', 'n/a', 'na') or genotype != 'WT':
        raise ValueError('Verified WT animal identity required: '+str(path))
    ids = load_asset_document('runtime_model_v1.json')['neuron_ids']
    animal = load_randi_animal(path, ids)
    if animal['subject'] != subject: raise ValueError('Animal ID disagreement')
    trials = animal_trials(animal, seed=0)
    if file_sha256(path) != before: raise ValueError('Raw input changed')
    return dict(path=str(path), sha256=before, subject=subject, genotype=genotype, trials=trials)


def fold_statistics(rows):
    from scripts.atlas_dataset_inventory import draft_folds
    from r_elegans.atlas_evaluation import pair_statistics
    subjects = [r['subject'] for r in rows]
    if len(set(subjects)) != len(subjects): raise ValueError('Repeated animal identity')
    trials = [t for r in rows for t in r['trials']]
    folds = draft_folds([t['subject'] for t in trials])
    if len(folds['folds']) != 5: raise ValueError('Five animal folds required')
    fold = folds['folds'][0]
    stats = {key: pair_statistics([t for t in trials if t['subject'] in set(names)])
             for key, names in [('train', fold['train_subjects']), ('heldout', fold['heldout_subjects'])]}
    return dict(stats=stats, subjects=fold, fivefold_membership=folds,
                accepted_trials=len(trials), raw_files=len(rows), usable_animals=len(set(t['subject'] for t in trials)))


def directions(codec):
    """Small initial panel of named groups; no claim of all-parameter coverage."""
    n = codec.size
    rows = []
    for name, columns in [('calcium_density', (6, 7, 8)), ('potassium_density', (0, 1, 2, 3, 4, 5, 15))]:
        x = np.zeros(n); x[:codec.sizes[0]] = np.isin(codec.channel[1], columns)
        rows.append((name, x))
    for name, start, stop in [('chemical', codec.sizes[0], sum(codec.sizes[:2])),
                              ('gap', sum(codec.sizes[:2]), n)]:
        x = np.zeros(n); x[start:stop] = 1.; rows.append((name, x))
    return rows


def train_choice(current, proposals):
    """Only finite, safe actual training improvements can be selected."""
    if not current['safe'] or not np.isfinite(current['train_loss']):
        raise ValueError('Unsafe/nonfinite current model cannot start training')
    choices = [current] + [p for p in proposals if p['safe'] and np.isfinite(p['train_loss'])]
    best = min(choices, key=lambda r: r['train_loss'])
    return best if best['train_loss'] < current['train_loss'] - 1e-10 else current


def worker(root, arm):
    manifest = json.loads((root/'manifest.json').read_text())
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,manifest['worker_cpus'][arm])
    import jax
    import jax.numpy as jnp
    from r_elegans.brain.conductance_network import load_network, ConductanceNetworkParams
    from r_elegans.brain.extrasynaptic import load_extrasynaptic_asset, initial_peptide_params, coupling_matrices
    from r_elegans.brain.peptide_state_replay import make_engine
    from scripts.stable_atlas_policy import Coordinates, loss
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    if not jax.config.x64_enabled or jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
        raise ValueError('Exactly one visible float64 GPU per arm required')
    folder = root/arm; folder.mkdir(exist_ok=True)
    capture_run_record(folder, REPO, dict(arm=arm), phase='E122-GPU-worker')
    parent = root/'parent.npz'
    if file_sha256(parent) != manifest['parent_sha256']: raise ValueError('Parent changed')
    for name,sha in manifest['source_sha256'].items():
        if file_sha256(REPO/name)!=sha:raise ValueError('Source changed before worker launch: '+name)
    with np.load(parent, allow_pickle=False) as z: arrays = {k:z[k] for k in z.files}
    codec = Coordinates(arrays); ids = arrays['neuron_ids'].tolist()
    _, fixed, config = load_network(parent)
    fixed = jax.tree.map(lambda v: jnp.asarray(v, dtype=jnp.float64), fixed)
    data = json.loads((root/'fold.json').read_text()); targets = manifest['targets']
    gs = np.zeros((len(targets)+2, len(ids)))
    gs[np.arange(len(targets)), [ids.index(t) for t in targets]] = manifest['stimulus_nS']
    asset = load_extrasynaptic_asset()
    if asset['neuron_ids'].tolist() != ids: raise ValueError('Peptide asset neuron order mismatch')
    peptide = initial_peptide_params(len(asset['sender']), dtype=jnp.float64)
    base_exc, base_inh = coupling_matrices(peptide, asset['sender'], asset['receiver'])
    if arm == 'network': base_exc, base_inh = base_exc*0., base_inh*0.
    engines = {False:make_engine(config, [ids.index('RMDL'), ids.index('RMDR')]),
               True:make_engine(config, [ids.index('RMDL'), ids.index('RMDR')], dt=.0005, step_ms=.015625)}
    history = []; counter = 0

    def evaluate(x, gains, *, fine=False, heldout=False):
        nonlocal counter
        before = time.monotonic(); name = f'eval-{counter:04d}'; counter += 1
        out = folder/name; out.mkdir()
        a = codec.apply(x)
        p = ConductanceNetworkParams(**{k:jnp.asarray(a['params_'+k]) for k in ConductanceNetworkParams._fields})
        exc, inh = base_exc*jnp.exp(gains[0]), base_inh*jnp.exp(gains[1])
        anchors, replay = engines[fine]
        start, phase = anchors(p, fixed, exc, inh, peptide.log_tau_s)
        means, final, trace = jax.tree.map(np.asarray, replay(p, fixed, exc, inh, peptide.log_tau_s, start, jnp.asarray(gs)))
        response = means[:, :len(targets)] - means[:, -1:, :]
        duplicate_error = float(np.max(abs(means[:, -1]-means[:, -2])))
        finite = bool(all(np.isfinite(v).all() for v in jax.tree.leaves((means, final, trace))))
        v = trace[0]; safe = bool(finite and v.min() >= -120 and v.max() <= 80 and
            final.neural.voltage_mV.min()>=-120 and final.neural.voltage_mV.max()<=80 and duplicate_error < 1e-12)
        rhythm=[rhythm_metrics(v[phase,-1,:,cell],.0005 if fine else .001) for phase in range(4) for cell in range(2)]
        safe=safe and all(r['passed'] for r in rhythm)
        score = loss(response, targets, ids, data, 'train')
        # This pilot's small prior is included in actual train-only comparisons.
        score += float(.001*np.mean((x/.1)**2) + .001*np.mean((gains/.1)**2))
        np.savez_compressed(out/'network.npz', **a)
        np.savez_compressed(out/'state.npz', **{'start_'+k:np.asarray(v) for k,v in start.neural._asdict().items()},
            **{'final_'+k:np.asarray(v) for k,v in final.neural._asdict().items()},
            start_peptide=np.asarray(start.peptide), offsets=x, peptide_log_gains=gains,
            response=response, rmd_voltage=trace[0], mean_peptide=trace[1], mean_indicator=trace[2],
            final_peptide=final.peptide, mean_indicator_by_cell=means)
        row = dict(name=name, fine=fine, train_loss=score, safe=safe, elapsed_seconds=time.monotonic()-before,
            heldout_loss=loss(response, targets, ids, data, 'heldout') if heldout else None,
            phase=phase, duplicate_sham_error=duplicate_error, parameter_audit=codec.audit(a,x),
            peptide_log_gains=gains.tolist(), peptide_bounds_passed=bool(np.max(abs(gains)) <= .1+1e-12),
            mean_peptide_level=float(np.mean(final.peptide)), maximum_peptide_level=float(np.max(final.peptide)),
            maximum_final_peptide_conductance_nS=float(config.synaptic_unit_nS*np.max(np.einsum('ij,...j->...i',np.asarray(exc+inh),final.peptide))),
            response_max=float(np.max(abs(response))), rmd_p2p_mV=np.ptp(v, axis=2).tolist(),
            short_window_rmd_checks=rhythm,
            sha256={str(out/f):file_sha256(out/f) for f in ('network.npz','state.npz')})
        write(out/'metrics.json',row); history.append(row); write(folder/'history.json',history)
        print(json.dumps(dict(arm=arm, **{k:row[k] for k in ('name','fine','train_loss','safe','elapsed_seconds')})),flush=True)
        return row

    x = np.zeros(codec.size); gains = np.zeros(2)
    current = evaluate(x,gains); baseline = current
    if not current['safe']: raise ValueError('Baseline short-window dynamics/control checks failed')
    schedule = directions(codec)
    best_x, best_gains = x.copy(), gains.copy(); updates = []
    np.savez_compressed(folder/'optimizer.npz',offsets=best_x,peptide_log_gains=best_gains,next_group=0)
    # Matched arms get at most four +/- group probes in this integration pilot.
    for name,direction in schedule[:4]:
        times = [r['elapsed_seconds'] for r in history]
        estimate = max(times[-3:])
        if time.time()+2.6*estimate > manifest['search_deadline_unix']: break
        candidates=[]; coordinates={}
        for sign in (-1,1):
            xx,gg=best_x.copy(),best_gains.copy()
            xx=np.clip(xx+sign*.01*direction,-.1,.1)
            r=evaluate(xx,gg); candidates.append(r);coordinates[r['name']]=(xx,gg)
        chosen=train_choice(current,candidates)
        updates.append(dict(group=name,before=current['name'],proposals=[r['name'] for r in candidates],selected=chosen['name']))
        if chosen['name']!=current['name']: best_x,best_gains=coordinates[chosen['name']]
        current=chosen;write(folder/'selection_history.json',updates)
        np.savez_compressed(folder/'optimizer.npz',offsets=best_x,peptide_log_gains=best_gains,next_group=len(updates))
    # Fresh selected endpoint repeat exposes the held-out score only after search.
    selected=evaluate(best_x,best_gains,heldout=True)
    refined=evaluate(best_x,best_gains,fine=True,heldout=True)
    with np.load(folder/baseline['name']/'state.npz') as z:
        baseline_heldout=loss(z['response'],targets,ids,data,'heldout')
    with np.load(folder/selected['name']/'state.npz') as a, np.load(folder/refined['name']/'state.npz') as b:
        delta=float(np.max(abs(a['response']-b['response'])))
        tolerance=float(1e-7+.1*np.max(abs(b['response'])))
    result=dict(arm=arm,completed=True,baseline=baseline,selected=selected,refined=refined,
        updates=updates,baseline_heldout_loss=baseline_heldout,changed_coordinates=int(np.count_nonzero(best_x)),numerical=dict(max_response_difference=delta,
        tolerance=tolerance,passed=delta<=tolerance),selection_reproduced=abs(selected['train_loss']-current['train_loss'])<1e-10,
        independent_cv_completed=False,behavior_qualified=False,unavailable_evidence=UNAVAILABLE)
    write(folder/'summary.json',result)


def audit(root):
    """Recompute selection, artifact hashes and parameter/frozen-field arithmetic."""
    from scripts.stable_atlas_policy import Coordinates,loss
    from r_elegans.brain.rhythm_qualification import rhythm_metrics
    problems=[];arms={}
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {}
    if not manifest: problems.append('Preparation/manifest incomplete')
    if manifest:
        with np.load(root/'parent.npz',allow_pickle=False) as z: codec=Coordinates({k:z[k] for k in z.files})
        ids=codec.origin['neuron_ids'].tolist();data=json.loads((root/'fold.json').read_text())
        subjects=data['subjects']
        if set(subjects['train_subjects']) & set(subjects['heldout_subjects']):problems.append('Animal leakage')
        for record in root.glob('**/run_records/*/source_record.json'):
            r=json.loads(record.read_text())
            for name,sha in r['source_sha256'].items():
                if file_sha256(record.parent/'source'/name)!=sha:problems.append('Source archive: '+name)
        for path,sha in manifest['input_sha256'].items():
            if not Path(path).is_file() or file_sha256(path)!=sha: problems.append('Changed/missing input: '+path)
        for name,sha in manifest['source_sha256'].items():
            if file_sha256(REPO/name)!=sha:problems.append('Source changed during run: '+name)
        for arm in ('network','peptide'):
            folder=root/arm;path=folder/'summary.json'
            s=json.loads(path.read_text()) if path.exists() else {}
            if not s:problems.append(arm+': incomplete')
            history=json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
            by={r['name']:r for r in history}
            unfinished=sorted(p.name for p in folder.glob('eval-*') if p.is_dir() and p.name not in by)
            if unfinished:problems.append(arm+': incomplete evaluation artifacts '+','.join(unfinished))
            for row in history:
                for p,sha in row['sha256'].items():
                    if file_sha256(p)!=sha: problems.append('Artifact hash: '+p)
                with np.load(folder/row['name']/'state.npz') as z, np.load(folder/row['name']/'network.npz') as n:
                    if not codec.audit({k:n[k] for k in n.files},z['offsets'])['passed']: problems.append('Parameter arithmetic: '+row['name'])
                    if np.max(abs(z['peptide_log_gains']))>.1+1e-12: problems.append('Peptide bounds: '+row['name'])
                    expected=loss(z['response'],manifest['targets'],ids,data,'train')+.001*np.mean((z['offsets']/.1)**2)+.001*np.mean((z['peptide_log_gains']/.1)**2)
                    if abs(expected-row['train_loss'])>1e-12:problems.append('Training score: '+row['name'])
                    if row['heldout_loss'] is not None and abs(loss(z['response'],manifest['targets'],ids,data,'heldout')-row['heldout_loss'])>1e-12:problems.append('Held-out score: '+row['name'])
                    means=z['mean_indicator_by_cell']
                    if np.max(abs(means[:,-1]-means[:,-2]))>=1e-12:problems.append('Duplicate sham: '+row['name'])
                    v=z['rmd_voltage'];checks=[rhythm_metrics(v[phase,-1,:,cell],.0005 if row['fine'] else .001) for phase in range(4) for cell in range(2)]
                    if checks!=row['short_window_rmd_checks']:problems.append('RMD arithmetic: '+row['name'])
                    finite=all(np.isfinite(z[k]).all() for k in z.files)
                    recomputed_safe=bool(finite and v.min()>=-120 and v.max()<=80 and z['final_voltage_mV'].min()>=-120 and
                        z['final_voltage_mV'].max()<=80 and np.max(abs(means[:,-1]-means[:,-2]))<1e-12 and all(c['passed'] for c in checks))
                    if recomputed_safe!=row['safe']:problems.append('Safety arithmetic: '+row['name'])
            for step in s.get('updates',[]):
                expected=train_choice(by[step['before']],[by[n] for n in step['proposals']])['name']
                if step['selected']!=expected:problems.append('Training-only selection: '+arm)
            if s:
                chosen=s['baseline']['name']
                for step in s['updates']:chosen=step['selected']
                repeat=abs(s['selected']['train_loss']-by[chosen]['train_loss'])<1e-10
                if repeat!=s['selection_reproduced']:problems.append('Selected endpoint reproduction: '+arm)
                with np.load(folder/s['selected']['name']/'state.npz') as a, np.load(folder/s['refined']['name']/'state.npz') as b:
                    delta=float(np.max(abs(a['response']-b['response'])));tol=float(1e-7+.1*np.max(abs(b['response'])))
                if delta!=s['numerical']['max_response_difference'] or (delta<=tol)!=s['numerical']['passed']:problems.append('Numerical arithmetic: '+arm)
            arms[arm]={k:s.get(k) for k in ('completed','changed_coordinates','numerical','selection_reproduced')}
            arms[arm]['all_evaluations_safe']=all(r['safe'] for r in history)
            arms[arm]['incomplete_evaluations']=unfinished
            arms[arm]['evaluations']=len(history)
            arms[arm]['evaluation_seconds']=[r['elapsed_seconds'] for r in history]
            if s:
                arms[arm]['train_loss_change']=s['selected']['train_loss']-s['baseline']['train_loss']
                arms[arm]['heldout_loss']=s['selected']['heldout_loss']
                arms[arm]['heldout_loss_change']=s['selected']['heldout_loss']-s['baseline_heldout_loss']
                arms[arm]['maximum_final_peptide_conductance_nS']=s['selected']['maximum_final_peptide_conductance_nS']
            times=[r['elapsed_seconds'] for r in history if not r['fine']][1:]
            arms[arm]['estimated_steady_evaluation_seconds']=float(np.median(times)) if times else None
    signal_comparison=None
    if len(arms)==2 and all(a['completed'] for a in arms.values()):
        summaries=[json.loads((root/arm/'summary.json').read_text()) for arm in ('network','peptide')]
        paths=[root/arm/s['baseline']['name']/'state.npz' for arm,s in zip(('network','peptide'),summaries)]
        with np.load(paths[0]) as a,np.load(paths[1]) as b:
            signal_comparison=dict(baseline_response_max_difference=float(np.max(abs(a['response']-b['response']))),
                baseline_sham_rmd_max_difference_mV=float(np.max(abs(a['rmd_voltage'][:,-1]-b['rmd_voltage'][:,-1]))),
                scope='Same wired prior, separately settled states with extrasynaptic term off/on; no behavior inference')
    result=dict(audit_complete=True,execution_complete=bool(len(arms)==2 and all(a['completed'] for a in arms.values())),integrity_passed=not problems,problems=problems,arms=arms,signal_comparison=signal_comparison,
        unavailable_evidence=UNAVAILABLE,long_run_ready=False,
        scope='Integration/throughput pilot only; no fivefold or behavioral qualification. Review audit before further training.')
    write(root/'audit.json',result);return result


def timing_report(root,result):
    rates={name:a.get('estimated_steady_evaluation_seconds') for name,a in result.get('arms',{}).items()}
    known={k:v for k,v in rates.items() if v is not None}
    report=dict(steady_evaluation_seconds=rates,scope='Small-pilot projections, not guaranteed training time or time to biological success',
        long_run_prerequisites=['Audit reviewed; no partial/integrity failures','Reproducible measured training descent',
            'Three-level numerical and sustained dynamics checks','Checkpoint restart drill before multi-day work',
            'Measured target batching/memory and disk-growth limits','Five independent fold fits; heldout scores never select updates'])
    if len(known)==2:
        report['estimated_hours_per_100_group_probes_both_arms']=2*100*max(known.values())/3600
        report['estimated_hours_per_100_probes_each_of_five_folds']=5*2*100*max(known.values())/3600
        report['projection_excludes']='Compilation, endpoint refinement, expanded target coverage, sustained assays and audits add time; gains may plateau.'
    write(root/'timing.json',report)


def coordinate(root):
    started=time.monotonic(); job=json.loads((root/'job.json').read_text())
    raw=Path(job['data_root'])/'raw/functional/randi2023_dandi001075_segmentation'
    paths=sorted(raw.glob('sub-*/*.nwb'))
    if not paths:raise ValueError('No raw NWB files')
    # Independent files on bounded CPU processes, not 128 competing BLAS threads.
    with ProcessPoolExecutor(max_workers=min(6,len(paths))) as pool: rows=list(pool.map(read_trials,map(str,paths)))
    data=fold_statistics(rows);write(root/'fold.json',data)
    write(root/'raw_inputs.json',[{k:v for k,v in row.items() if k!='trials'} for row in rows])
    from scripts.stable_atlas_policy import eligible_targets
    pool=eligible_targets(data,'train')
    if len(pool)<4:raise ValueError('Four eligible training targets required')
    rng=np.random.default_rng(SEED)
    targets=sorted(rng.choice(pool,4,replace=False).tolist())
    parent=REPO/'r_elegans/assets/mac_rhythm_v1_polarized/network.npz'
    if file_sha256(parent)!=REFERENCE_SHA:raise ValueError('Bundled reference hash changed; lineage review required')
    shutil.copy2(parent,root/'parent.npz')
    cpus=physical_cpus()
    if len(cpus)<8:raise ValueError('Eight available physical cores required for two disjoint workers')
    manifest=dict(experiment='E122',seed=SEED,targets=targets,stimulus_nS=.03,
        parent_sha256=file_sha256(parent),parent=str(parent),
        target_rng_state=rng.bit_generator.state,
        source_sha256={str(p.relative_to(REPO)):file_sha256(p) for p in
            [REPO/'scripts/run_cv_peptide_pilot.py',REPO/'scripts/stable_atlas_policy.py',
             REPO/'r_elegans/brain/peptide_state_replay.py',REPO/'r_elegans/brain/conductance_network.py',
             REPO/'r_elegans/brain/single_compartment.py',REPO/'r_elegans/brain/extrasynaptic.py',
             REPO/'r_elegans/assets/extrasynaptic_v1.npz']},
        parent_scope='Bundled pre-atlas reference, shared AWC/RMD priors with earlier rhythm engineering; not E120. No claim of a pristine unseen dataset.',
        search_deadline_unix=time.time()+max(0,30*60-(time.monotonic()-started)),
        input_sha256={str(parent):file_sha256(parent),str(root/'parent.npz'):file_sha256(root/'parent.npz'),
            str(root/'fold.json'):file_sha256(root/'fold.json'),str(root/'raw_inputs.json'):file_sha256(root/'raw_inputs.json'),
            **{r['path']:r['sha256'] for r in rows}},
        precision='float64',fold=0,training_only_selection=True,
        peptide_scope='269 expression-defined pairs with fixed equal prior gains and shared 5s memory. Wired/channel groups fitted jointly in peptide arm; peptide gain fitting deferred.',
        worker_cpus=dict(network=cpus[:4],peptide=cpus[4:8]),available_physical_cores=len(cpus),
        total_minutes=60,search_target_minutes=30,final_check_reserve_minutes=24,audit_cleanup_reserve_minutes=6)
    write(root/'manifest.json',manifest)
    processes=[];logs=[]
    try:
        for gpu,arm in enumerate(('network','peptide')):
            log=(root/(arm+'.log')).open('w');logs.append(log)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true',
                XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            processes.append(subprocess.Popen([sys.executable,'-u','-m','scripts.run_cv_peptide_pilot','--output-dir',str(root),'--worker',arm],env=env,stdout=log,stderr=subprocess.STDOUT))
        write(root/'workers.json',dict(pids=[p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            print(json.dumps(dict(elapsed_minutes=(time.monotonic()-started)/60,running=[a for a,p in zip(('network','peptide'),processes) if p.poll() is None])),flush=True)
            try:
                r=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                with (root/'gpu_telemetry.jsonl').open('a') as f:f.write(json.dumps(dict(time_unix=time.time(),returncode=r.returncode,rows=r.stdout.splitlines()))+'\n')
            except (OSError,subprocess.TimeoutExpired):pass
            time.sleep(15)
        write(root/'worker_exit_codes.json',{arm:p.returncode for arm,p in zip(('network','peptide'),processes)})
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/home/carsonc/r-elegans-data'))
    p.add_argument('--worker',choices=('network','peptide'));p.add_argument('--coordinate',action='store_true');p.add_argument('--report',action='store_true')
    a=p.parse_args();root=a.output_dir.resolve()
    if a.report:
        print('Run:',root)
        audit_path=root/'audit.json'
        if audit_path.exists():
            r=json.loads(audit_path.read_text())
            print('Audit:',{k:r.get(k) for k in ('audit_complete','execution_complete','integrity_passed','long_run_ready','problems')})
            for arm,s in r.get('arms',{}).items():
                print(arm,{k:s.get(k) for k in ('completed','evaluations','changed_coordinates','train_loss_change',
                    'heldout_loss','heldout_loss_change','numerical','selection_reproduced','estimated_steady_evaluation_seconds')})
            print('Unavailable evidence:',len(r.get('unavailable_evidence',[])),'items; full audit:',audit_path)
        else:print('Audit not yet available; inspect adjacent .log and network.log/peptide.log')
        for f in ('budget_receipt.json','timing.json'):
            print(f,json.dumps(json.loads((root/f).read_text())) if (root/f).exists() else 'Not yet available')
        return
    if a.worker:return worker(root,a.worker)
    if a.coordinate:
        try:return coordinate(root)
        except Exception as exc:
            write(root/'failure.json',dict(error=repr(exc)));raise
    if root==REPO or REPO in root.parents or root in REPO.parents: p.error('External output required')
    root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    if shutil.disk_usage(root).free<5*1024**3:raise ValueError('At least5GiB free required for pilot and source records')
    write(root/'job.json',dict(data_root=str(a.data_root.resolve()),total_minutes=60,seed=SEED))
    capture_run_record(root,REPO,vars(a),phase='E122-one-hour-integration-pilot')
    from scripts.total_job_budget import run_group
    code,timed_out=run_group([sys.executable,'-u','-m','scripts.run_cv_peptide_pilot','--output-dir',str(root),'--coordinate'],max(.01,54*60-(time.monotonic()-started)))
    try:result=audit(root)
    except Exception as exc:
        traceback.print_exc();result=dict(audit_complete=False,integrity_passed=False,problems=[repr(exc)],execution_complete=False)
        write(root/'audit.json',result)
    timing_report(root,result)
    receipt=dict(exit_code=code,timed_out=timed_out,elapsed_seconds=time.monotonic()-started,total_minutes=60,
        all_steps_finished=bool(code==0 and result['execution_complete'] and result['audit_complete']),
        audit_complete=result['audit_complete'],long_run_ready=False,training_qualified=False)
    write(root/'budget_receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return 0 if receipt['all_steps_finished'] and result['integrity_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
